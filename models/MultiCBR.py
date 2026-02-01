#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
try:
    from torch_scatter import scatter
except ImportError:
    scatter = None

def scatter_sum(src, index, dim_size=None):
    if scatter is not None:
        return scatter(src, index, dim=0, dim_size=dim_size, reduce='add')
    else:
        # Pure PyTorch implementation
        if dim_size is None:
            dim_size = index.max().item() + 1
        out = torch.zeros((dim_size, src.size(1)), dtype=src.dtype, device=src.device)
        return out.index_add_(0, index, src)


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class GraphConv_CA(nn.Module):
    """
    Collaborative Adaptive Graph Convolutional Network (from CAGCN)
    """
    def __init__(self, num_layers):
        super(GraphConv_CA, self).__init__()
        self.num_layers = num_layers

    def forward(self, embed, edge_index, trend):
        # embed: [n_nodes, channel]
        # edge_index: [2, n_edges]
        # trend: [n_edges] (CIR weights)
        
        agg_embed = embed
        embs = [embed]
        
        row, col = edge_index
        n_nodes = embed.shape[0]

        for hop in range(self.num_layers):
            # Message Passing:
            # out[e] = embed[row[e]] * trend[e]
            out = agg_embed[row] * trend.unsqueeze(-1)
            
            # Aggregation: sum messages to destination node (col)
            agg_embed = scatter_sum(out, col, dim_size=n_nodes)
            
            embs.append(F.normalize(agg_embed, p=2, dim=1)) # Normalize as in MultiCBR

        return embs


class MultiCBR(nn.Module):
    def __init__(self, conf, raw_graph, trends=None):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]
        self.trend_coeff = conf.get("trend_coeff", 1.0)

        self.fusion_weights = conf['fusion_weights']

        self.init_emb()
        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        
        # Load CIR trends if provided
        self.trends = trends
        if self.trends is not None:
            self.trend_ub, self.trend_ui, self.trend_bi = [t.to(device) for t in self.trends]
            
            # Extract indices and values for CAGCN propagation
            self.ub_indices = self.trend_ub.indices()
            self.ub_values = self.trend_ub.values() * self.trend_coeff
            
            self.ui_indices = self.trend_ui.indices()
            self.ui_values = self.trend_ui.values() * self.trend_coeff
            
            self.bi_indices = self.trend_bi.indices()
            self.bi_values = self.trend_bi.values() * self.trend_coeff
        else:
            # Fallback to original logic if trends not provided (or error)
            raise ValueError("CIR trends must be provided for CAGCN mode")

        # CAGCN* Encoders for each view
        self.encoder_ub = GraphConv_CA(self.num_layers)
        self.encoder_ui = GraphConv_CA(self.num_layers)
        self.encoder_bi = GraphConv_CA(self.num_layers)
        
        # Initialize propagation graphs (default: full graph without dropout)
        # We need these to be available even if ED_drop=False (e.g. first epoch or eval)
        self.UB_propagation_graph = to_tensor(self.ub_graph).to(self.device)
        self.UI_propagation_graph = to_tensor(self.ui_graph).to(self.device)
        self.BI_propagation_graph = to_tensor(self.bi_graph).to(self.device)

        if self.conf['aug_type'] == 'MD':
            self.init_md_dropouts()
        elif self.conf['aug_type'] == "Noise":
            self.init_noise_eps()


    def init_md_dropouts(self):
        self.UB_dropout = nn.Dropout(self.conf["UB_ratio"], True)
        self.UI_dropout = nn.Dropout(self.conf["UI_ratio"], True)
        self.BI_dropout = nn.Dropout(self.conf["BI_ratio"], True)
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout
        }


    def init_noise_eps(self):
        self.UB_eps = self.conf["UB_ratio"]
        self.UI_eps = self.conf["UI_ratio"]
        self.BI_eps = self.conf["BI_ratio"]
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def init_fusion_weights(self):
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"

        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1),\
            "The number of layer fusion weights does not correspond to number of layers"

        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)


    def propagate_cagcn(self, encoder, A_feature, B_feature, indices, values, graph_type, layer_coef, test, original_graph=None):
        # Concatenate features: [A; B]
        features = torch.cat((A_feature, B_feature), 0)
        
        # 1. CAGCN Trend Propagation
        # all_features_list is [embed_0, embed_1, ...]
        trend_features_list = encoder(features, indices, values)
        
        # 2. Original Graph Propagation (LightGCN style)
        # If original_graph is provided, we use it to propagate as well.
        # This ensures we don't lose the collaborative signal.
        # Original graph is usually normalized adjacency.
        
        final_features_list = []
        
        # We need to manually propagate on original graph if provided
        if original_graph is not None:
            curr_features = features
            origin_features_list = [curr_features]
            for i in range(self.num_layers):
                # spmm: sparse matrix multiplication
                # original_graph is torch.sparse_coo_tensor
                curr_features = torch.sparse.mm(original_graph, curr_features)
                origin_features_list.append(F.normalize(curr_features, p=2, dim=1))
                
            # Fuse: Origin + Trend
            # Note: trend_features_list[0] is just input features, same as origin_features_list[0]
            for i in range(len(trend_features_list)):
                # Simple addition: Origin + Trend
                # Since values in trend already multiplied by trend_coeff
                fused = origin_features_list[i] + trend_features_list[i]
                final_features_list.append(fused)
        else:
            # Fallback (should not happen in this fix)
            final_features_list = trend_features_list

        # Layer Aggregation (MultiCBR logic)
        # layer_coef: [1, 1, 1, num_layers+1]
        # We stack features: [num_layers+1, n_nodes, emb_size]
        all_features = torch.stack(final_features_list, dim=0)
        
        # Weighted sum of layers
        # layer_coef is [1, 1, 1, L+1], broadcast to [L+1, N, D]
        # permute layer_coef to [L+1, 1, 1] for broadcasting?
        # In init: self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1) -> [1, L+1, 1]
        # Wait, init says: unsqueeze(0).unsqueeze(-1). 
        # fusion_weights['UB_layer'] length is L+1.
        # So shape is [1, L+1, 1].
        # We need to permute features to [1, L+1, N, D] or just sum over dim 0?
        
        # MultiCBR original logic likely sums:
        # agg_feature = sum(feat[i] * w[i])
        
        # Let's check init again:
        # self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1) -> [1, L+1, 1] ??
        # No, UB_layer_coefs is 1D tensor of size L+1.
        # unsqueeze(0) -> [1, L+1]
        # unsqueeze(-1) -> [1, L+1, 1]
        # This seems designed for [Batch, Layer, Emb] ? No.
        
        # Let's assume standard weighted sum:
        # all_features: [L+1, N, D]
        # coef: [1, L+1, 1] -> squeeze(0) -> [L+1, 1]
        
        coef = layer_coef.squeeze(0) # [L+1, 1]
        
        # Weighted sum along layer dimension (0)
        # all_features * coef: broadcasting [L+1, N, D] * [L+1, 1, 1]
        out = torch.sum(all_features * coef.unsqueeze(-1), dim=0)
        
        # Split back to A and B
        dim_A = A_feature.shape[0]
        A_out = out[:dim_A]
        B_out = out[dim_A:]
        
        return A_out, B_out


    def aggregate_cagcn(self, node_feature, indices, values, target_dim, source_dim, graph_type, test):
        # Aggregate from Source -> Target
        # indices is symmetric [ (r1, c1), (r2, c2) ... ] for the full bipartite graph
        # We need the block that maps Source indices to Target indices.
        # e.g. BI aggregation: Items(Source) -> Bundles(Target).
        # In BI graph: Rows=Bundles, Cols=Items.
        # Full matrix: [[0, BI], [BI.T, 0]].
        # BI block: Row range [0, n_B], Col range [n_B, n_B+n_I].
        # We want to aggregate FROM cols TO rows.
        
        # For simplicity, we can filter edges where row < target_dim and col >= target_dim
        # But this filtering is slow every time.
        # Alternatively, we can assume the symmetric indices handle full propagation,
        # so we can just propagate one step on the full graph, and take the Target part.
        
        # Construct full feature vector: [Target_Zero; Source_Feature]
        # Then propagate 1 step. Result[0:target_dim] is the aggregation.
        
        zeros = torch.zeros(target_dim, node_feature.shape[1]).to(self.device)
        full_feature = torch.cat([zeros, node_feature], 0)
        
        row, col = indices
        
        # One step propagation
        # out[e] = feature[row[e]] * value[e]
        # agg[c] = sum(out[row==...])
        # We use scatter sum
        out = full_feature[row] * values.unsqueeze(-1)
        agg_full = scatter_sum(out, col, dim_size=target_dim + source_dim)
        
        aggregated_feature = agg_full[:target_dim]
        
        # Apply Dropout/Noise
        if self.conf["aug_type"] == "MD" and not test:
            mess_dropout = self.mess_dropout_dict[graph_type]
            aggregated_feature = mess_dropout(aggregated_feature)
        elif self.conf["aug_type"] == "Noise" and not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            eps = self.eps_dict[graph_type]
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature


    def fuse_users_bundles_feature(self, users_feature, bundles_feature):
        users_feature = torch.stack(users_feature, dim=0)
        bundles_feature = torch.stack(bundles_feature, dim=0)

        # Modal aggregation
        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        return users_rep, bundles_rep


    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        UB_users_feature, UB_bundles_feature = self.propagate_cagcn(
            self.encoder_ub, self.users_feature, self.bundles_feature, 
            self.ub_indices, self.ub_values, "UB", self.UB_layer_coefs, test,
            original_graph=self.UB_propagation_graph
        )

        #  =============================  UI graph propagation  =============================
        UI_users_feature, UI_items_feature = self.propagate_cagcn(
            self.encoder_ui, self.users_feature, self.items_feature, 
            self.ui_indices, self.ui_values, "UI", self.UI_layer_coefs, test,
            original_graph=self.UI_propagation_graph
        )
        
        # Aggregate Items -> Bundles (using BI graph structure)
        # BI graph: Bundles (Rows), Items (Cols)
        UI_bundles_feature = self.aggregate_cagcn(
            UI_items_feature, self.bi_indices, self.bi_values, 
            self.num_bundles, self.num_items, "BI", test
        )

        #  =============================  BI graph propagation  =============================
        BI_bundles_feature, BI_items_feature = self.propagate_cagcn(
            self.encoder_bi, self.bundles_feature, self.items_feature, 
            self.bi_indices, self.bi_values, "BI", self.BI_layer_coefs, test,
            original_graph=self.BI_propagation_graph
        )
        
        # Aggregate Items -> Users (using UI graph structure)
        # UI graph: Users (Rows), Items (Cols)
        BI_users_feature = self.aggregate_cagcn(
            BI_items_feature, self.ui_indices, self.ui_values, 
            self.num_users, self.num_items, "UI", test
        )

        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]

        users_rep, bundles_rep = self.fuse_users_bundles_feature(users_feature, bundles_feature)

        return users_rep, bundles_rep


    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, users_feature, bundles_feature):
        # users_feature / bundles_feature: [bs, 1+neg_num, emb_size]
        pred = torch.sum(users_feature * bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        # cl is abbr. of "contrastive loss"
        u_view_cl = self.cal_c_loss(users_feature, users_feature)
        b_view_cl = self.cal_c_loss(bundles_feature, bundles_feature)

        c_losses = [u_view_cl, b_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss


    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_rep, bundles_rep = self.get_multi_modal_representations()

        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        scores = torch.mm(users_feature[users], bundles_feature.t())
        return scores

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 


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


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


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


class MultiCBR(nn.Module):
    def __init__(self, conf, raw_graph):
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

        self.fusion_weights = conf['fusion_weights']
        
        # CAGCN settings
        self.trend_coeff = conf.get("trend_coeff", 1.0)
        self.trend_mix = conf.get("trend_mix", False)
        self.trend_norm = conf.get("trend_norm", "row")
        self.trend_mix_layers = conf.get("trend_mix_layers", -1)
        self.trend_topk = conf.get("trend_topk", 0)
        
        self.trend_coeff_ub = conf.get("trend_coeff_ub", self.trend_coeff)
        self.trend_coeff_ui = conf.get("trend_coeff_ui", self.trend_coeff)
        self.trend_coeff_bi = conf.get("trend_coeff_bi", self.trend_coeff)

        # view-wise switches (default: follow global trend_mix)
        self.trend_mix_ub = conf.get("trend_mix_ub", conf.get("trend_mix", False))
        self.trend_mix_ui = conf.get("trend_mix_ui", conf.get("trend_mix", False))
        self.trend_mix_bi = conf.get("trend_mix_bi", conf.get("trend_mix", False))

        self.init_emb()
        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        
        # CAGCN: Load trends
        if "trends" in self.conf and self.conf["trends"] is not None:
            self.trend_ub, self.trend_ui, self.trend_bi = self.conf["trends"]
            self.trend_ub = self.trend_ub.to(self.device)
            self.trend_ui = self.trend_ui.to(self.device)
            self.trend_bi = self.trend_bi.to(self.device)

            # Normalize trends if mixed propagation is enabled or requested
            if self.trend_norm == 'row':
                 if self.trend_ub is not None: self.trend_ub = self.row_normalize_sparse(self.trend_ub)
                 if self.trend_ui is not None: self.trend_ui = self.row_normalize_sparse(self.trend_ui)
                 if self.trend_bi is not None: self.trend_bi = self.row_normalize_sparse(self.trend_bi)
                 
            # ===== DEBUG LOG (add) ===== 
            if getattr(self, "trend_norm", None) == "row": 
                print("[TrendNorm] row-normalize applied") 
            # ===== DEBUG LOG (end) ===== 
                 
        else:
            self.trend_ub = None
            self.trend_ui = None
            self.trend_bi = None
            
        # ===== DEBUG LOG (add) ===== 
        def _sp_info(t, name): 
            if t is None: 
                return f"{name}=None" 
            tt = t.coalesce() 
            return f"{name}: shape={tuple(tt.shape)} nnz={tt._nnz()} device={tt.device} dtype={tt.dtype}" 
        
        print("[TrendLoad]", 
              _sp_info(self.trend_ub, "trend_ub"), 
              _sp_info(self.trend_ui, "trend_ui"), 
              _sp_info(self.trend_bi, "trend_bi")) 
        print("[MixCfg]", 
              "trend_mix=", conf.get("trend_mix", False), 
              "trend_norm=", conf.get("trend_norm", "row"), 
              "trend_mix_layers=", conf.get("trend_mix_layers", -1), 
              "mix_ub/ui/bi=", self.trend_mix_ub, self.trend_mix_ui, self.trend_mix_bi, 
              "coeff_ub/ui/bi=", 
              conf.get("trend_coeff_ub", conf.get("trend_coeff", 1.0)), 
              conf.get("trend_coeff_ui", conf.get("trend_coeff", 1.0)), 
              conf.get("trend_coeff_bi", conf.get("trend_coeff", 1.0))) 
        # ===== DEBUG LOG (end) =====

        # generate the graph without any dropouts for testing
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)

        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)

        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

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


    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(device)


    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bipartite_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)


    def row_normalize_sparse(self, t):
        # t: sparse tensor
        t = t.coalesce()
        indices = t.indices()
        values = t.values()
        row = indices[0]
        num_rows = t.size(0)
        
        # Calculate row sum
        # Using scatter_add logic manually if no torch_scatter
        row_sum = torch.zeros(num_rows, device=t.device)
        row_sum.index_add_(0, row, values)
        
        # Inverse row sum
        inv = 1.0 / (row_sum + 1e-12)
        
        # Apply normalization
        new_values = values * inv[row]
        
        return torch.sparse_coo_tensor(indices, new_values, t.size()).coalesce()


    def propagate_mixed(self, A, T, features, num_layers, alpha, mask=None, mix_layers=-1):
        # A: Augmentated Graph (or Original)
        # T: Normalized Trend Graph
        
        if T is not None and features.device != T.device:
             T = T.to(features.device)
             
        # Check shapes if T is provided
        if T is not None:
             assert A.size() == T.size(), f"Graph shape mismatch: A {A.size()} vs T {T.size()}"

        all_features = [features]
        H = features
        
        for l in range(num_layers):
            # 1. Original Propagation
            H_A = torch.spmm(A, H)
            
            # Apply Augmentation to H_A if needed (already done in A construction usually)
            # But wait, propagate() does augmentation inside loop for MD/Noise
            if self.conf["aug_type"] == "MD" and mask is not None:
                # Mask logic for Message Dropout
                H_A = mask(H_A)
            elif self.conf["aug_type"] == "Noise" and mask is not None:
                # Noise logic
                # random_noise = torch.rand_like(H_A).to(self.device)
                random_noise = torch.rand(H_A.size(), device=self.device) # Avoid explicit like if possible, but shape is same
                eps = mask # mask is eps here
                H_A += torch.sign(H_A) * F.normalize(random_noise, dim=-1) * eps

            # 2. Mixed Propagation
            if self.trend_mix and T is not None and (mix_layers == -1 or l < mix_layers):
                H_T = torch.spmm(T, H)
                H = (1 - alpha) * H_A + alpha * H_T
            else:
                H = H_A
            
            H = F.normalize(H, p=2, dim=1)
            all_features.append(H)
            
        all_features = torch.stack(all_features, 1)
        # Apply layer coefficients? self.propagate does it at end
        # But layer_coef is passed to propagate...
        # Let's handle it outside or pass it in? 
        # propagate() does: all_features * layer_coef -> sum
        
        return all_features


    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        # Legacy propagate modified to use propagate_mixed logic if trend_mix is False
        # But actually we want to replace calls to propagate() with calls to a unified function
        # Let's keep propagate() for legacy support or non-mixed calls
        
        # ... (Existing propagate code) ...
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:
                mess_dropout = self.mess_dropout_dict[graph_type]
                features = mess_dropout(features)
            elif self.conf["aug_type"] == "Noise" and not test:
                random_noise = torch.rand_like(features).to(self.device)
                eps = self.eps_dict[graph_type]
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps

            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature
    
    
    def propagate_trend(self, trend_graph, A_feature, B_feature):
        # CAGCN Plugin: Propagate using trend graph (Simple GCN)
        if trend_graph is None:
            return torch.zeros_like(A_feature), torch.zeros_like(B_feature)
            
        features = torch.cat((A_feature, B_feature), 0)
        # One layer propagation for trend
        # Ensure device match
        if features.device != trend_graph.device:
            trend_graph = trend_graph.to(features.device)
            
        trend_features = torch.spmm(trend_graph, features)
        # Normalize?
        trend_features = F.normalize(trend_features, p=2, dim=1)
        
        A_trend, B_trend = torch.split(trend_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_trend, B_trend


    def aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        # simple embedding dropout on bundle embeddings
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
        if not hasattr(self, "_mix_logged"): 
            self._mix_logged = {"UB": False, "UI": False, "BI": False}
            
        # Helper to get mask/eps for augmentation
        def get_aug_mask(graph_type):
            if test: return None
            if self.conf["aug_type"] == "MD":
                return self.mess_dropout_dict[graph_type]
            elif self.conf["aug_type"] == "Noise":
                return self.eps_dict[graph_type]
            return None

        #  =============================  UB graph propagation  =============================
        if test:
            A_ub = self.UB_propagation_graph_ori
        else:
            A_ub = self.UB_propagation_graph
            
        if self.trend_mix and self.trend_mix_ub:
            # ===== DEBUG LOG (add) ===== 
            if not self._mix_logged["UB"]:
                print("[MixRun][UB] ACTIVE", 
                      "mix_layers=", self.trend_mix_layers, 
                      "alpha=", self.trend_coeff_ub) 
                self._mix_logged["UB"] = True
            # ===== DEBUG LOG (end) ===== 
            
            # Mixed Propagation
            features_ub = torch.cat((self.users_feature, self.bundles_feature), 0)
            all_feats = self.propagate_mixed(A_ub, self.trend_ub, features_ub, self.num_layers, self.trend_coeff_ub, get_aug_mask("UB"), self.trend_mix_layers)
            all_feats = all_feats * self.UB_layer_coefs
            all_feats = torch.sum(all_feats, dim=1)
            UB_users_feature, UB_bundles_feature = torch.split(all_feats, (self.users_feature.shape[0], self.bundles_feature.shape[0]), 0)
        else:
            # Legacy Propagation
            UB_users_feature, UB_bundles_feature = self.propagate(A_ub, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
            # Legacy Residual Plugin
            if self.trend_ub is not None:
                UB_trend_u, UB_trend_b = self.propagate_trend(self.trend_ub, self.users_feature, self.bundles_feature)
                UB_users_feature = UB_users_feature + self.trend_coeff * UB_trend_u
                UB_bundles_feature = UB_bundles_feature + self.trend_coeff * UB_trend_b

        #  =============================  UI graph propagation  =============================
        if test:
            A_ui = self.UI_propagation_graph_ori
            Agg_ui = self.BI_aggregation_graph_ori
        else:
            A_ui = self.UI_propagation_graph
            Agg_ui = self.BI_aggregation_graph

        if self.trend_mix and self.trend_mix_ui:
             if not self._mix_logged["UI"]:
                 print("[MixRun][UI] ACTIVE", 
                       "mix_layers=", self.trend_mix_layers, 
                       "alpha=", self.trend_coeff_ui) 
                 self._mix_logged["UI"] = True
                 
             features_ui = torch.cat((self.users_feature, self.items_feature), 0)
             all_feats = self.propagate_mixed(A_ui, self.trend_ui, features_ui, self.num_layers, self.trend_coeff_ui, get_aug_mask("UI"), self.trend_mix_layers)
             all_feats = all_feats * self.UI_layer_coefs
             all_feats = torch.sum(all_feats, dim=1)
             UI_users_feature, UI_items_feature = torch.split(all_feats, (self.users_feature.shape[0], self.items_feature.shape[0]), 0)
             UI_bundles_feature = self.aggregate(Agg_ui, UI_items_feature, "BI", test)
        else:
            UI_users_feature, UI_items_feature = self.propagate(A_ui, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(Agg_ui, UI_items_feature, "BI", test)
            
            # Legacy Residual Plugin
            if self.trend_ui is not None:
                UI_trend_u, UI_trend_i = self.propagate_trend(self.trend_ui, self.users_feature, self.items_feature)
                UI_users_feature = UI_users_feature + self.trend_coeff * UI_trend_u

        #  =============================  BI graph propagation  =============================
        if test:
            A_bi = self.BI_propagation_graph_ori
            Agg_bi = self.UI_aggregation_graph_ori
        else:
            A_bi = self.BI_propagation_graph
            Agg_bi = self.UI_aggregation_graph
            
        if self.trend_mix and self.trend_mix_bi:
             if not self._mix_logged["BI"]:
                 print("[MixRun][BI] ACTIVE", 
                       "mix_layers=", self.trend_mix_layers, 
                       "alpha=", self.trend_coeff_bi) 
                 self._mix_logged["BI"] = True
                 
             features_bi = torch.cat((self.bundles_feature, self.items_feature), 0)
             all_feats = self.propagate_mixed(A_bi, self.trend_bi, features_bi, self.num_layers, self.trend_coeff_bi, get_aug_mask("BI"), self.trend_mix_layers)
             all_feats = all_feats * self.BI_layer_coefs
             all_feats = torch.sum(all_feats, dim=1)
             BI_bundles_feature, BI_items_feature = torch.split(all_feats, (self.bundles_feature.shape[0], self.items_feature.shape[0]), 0)
             BI_users_feature = self.aggregate(Agg_bi, BI_items_feature, "UI", test)
        else:
            BI_bundles_feature, BI_items_feature = self.propagate(A_bi, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(Agg_bi, BI_items_feature, "UI", test)
            
            # Legacy Residual Plugin
            if self.trend_bi is not None:
                BI_trend_b, BI_trend_i = self.propagate_trend(self.trend_bi, self.bundles_feature, self.items_feature)
                BI_bundles_feature = BI_bundles_feature + self.trend_coeff * BI_trend_b

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

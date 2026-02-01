#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    if os.environ.get("MULTICBR_QUIET_STATS") == "1":
        return
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:                                                          
                all_bundles.append(i)                                                                                                   
                if len(all_bundles) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)


    def __len__(self):
        return len(self.u_b_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_pairs, b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        # Calculate CIR trends for CAGCN
        cagcn_type = conf.get('cagcn_type', 'jc')
        print(f"Calculating CIR trends ({cagcn_type})...")
        trend_ub = self.get_cir_trend(u_b_graph_train, cagcn_type, 'ub')
        trend_ui = self.get_cir_trend(u_i_graph, cagcn_type, 'ui')
        trend_bi = self.get_cir_trend(b_i_graph, cagcn_type, 'bi')
        self.trends = [trend_ub, trend_ui, trend_bi]

        default_workers_train = 0 if os.name == "nt" else 10
        default_workers_test = 0 if os.name == "nt" else 20
        num_workers_train = int(conf.get("num_workers_train", default_workers_train))
        num_workers_test = int(conf.get("num_workers_test", default_workers_test))

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=num_workers_train, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=num_workers_test)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=num_workers_test)


    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]


    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_pairs, b_i_graph


    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph


    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph

    def get_cir_trend(self, graph, type, name):
        # graph: scipy.sparse.csr_matrix (rows, cols)
        # type: jc, sc, lhn, co
        # name: ub, ui, bi (for cache file naming)
        
        path = os.path.join(self.path, self.name, 'trend_{}_{}.pt'.format(name, type))
        if os.path.exists(path):
            print(f'Loading precomputed trend for {name} ({type}) from {path}')
            return torch.load(path)
            
        print(f'Calculating trend for {name} ({type})...')
        
        # Convert to torch tensor for GPU calculation if possible
        # Since we have 24G GPU, we can try to do it on GPU with dense matrix if size permits
        # Or use sparse matrix operations on CPU/GPU
        
        # Here we implement Matrix-Formed calculation as in CAGCN
        # graph shape: [N_row, N_col]
        # We need to calculate weights for the full bipartite graph:
        # [[0, R], [R.T, 0]]
        # The output trend should be a sparse tensor corresponding to the edges in the bipartite graph
        
        # 1. Construct dense interaction matrix (if possible) or use sparse multiplication
        # Given the dataset size (NetEase: 18K users, 22K bundles, 123K items)
        # 123K * 123K float32 dense matrix is ~60GB, so we CANNOT form the full item-item matrix densely even on 24G GPU
        # We must stick to sparse operations or batch-wise dense operations.
        # CAGCN's 'co_ratio_deg_user_jacard' uses dense matrix which might OOM for items.
        # However, we only need the values at existing edges (Trend * Adjacency).
        # But CAGCN's implementation actually computes the similarity between users (based on items) 
        # and items (based on users), then assigns these similarities as edge weights.
        # Wait, CAGCN's code: 
        # edge_weight[users, i + n_users] = jacard_simi
        # It assigns the user-user similarity (or user-item?) let's re-read CAGCN logic carefully.
        
        # Re-reading CAGCN utils.py:
        # for i in range(n_items): 
        #    users = user_item_graph[:, i].nonzero() ...
        #    items = user_item_graph[users] ...
        #    user_user_cap = items @ items.t() ...
        #    jacard_simi = ...
        #    edge_weight[users, i + n_users] = jacard_simi
        
        # It calculates, for each item i, the similarity between the users who interacted with i.
        # Wait, "user_user_cap" is (num_users_interacted_with_i x num_users_interacted_with_i).
        # Then "jacard_simi = ... .mean(dim=1)".
        # It means for a specific edge (u, i), the weight is the average Jaccard similarity 
        # between u and ALL OTHER users who also interacted with i.
        # This is "Collaboration-Aware": how much u is similar to the group of users who bought i.
        
        # Let's implement this logic using sparse matrices to save memory.
        # Or since we know we have 24G GPU, maybe we can do it batch-wise.
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} for CIR calculation")
        
        # Convert scipy sparse to torch sparse
        coo = graph.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)).to(device)
        values = torch.from_numpy(coo.data.astype(np.float32)).to(device)
        shape = coo.shape
        
        # We need fast row/col access. 
        # Let's use dense if dimension < 20000, else sparse-based logic.
        # NetEase: Users ~18K, Bundles ~22K -> Can use dense for User-User sim, Bundle-Bundle sim?
        # 22000^2 * 4 bytes = ~1.9GB. This is fine!
        # Items ~123K -> Item-Item sim is too big.
        
        # Actually CAGCN logic is:
        # For edge (u, v):
        # 1. Find N(v) = {u', u'', ...} (users who interacted with item v)
        # 2. Calculate Sim(u, u') for all u' in N(v)
        # 3. Weight(u, v) = Average(Sim(u, u'))
        
        # So we need User-User Similarity Matrix (S_U) and Item-Item Similarity Matrix (S_I).
        # If we have S_U (N_u x N_u), then for edge (u, v), w = mean(S_U[u, N(v)]).
        
        # Let's implement Jaccard/SC/LHN/CO for generic bipartite graph (Row-Col).
        # Row nodes: U, Col nodes: V
        # We need S_Row (U x U) and S_Col (V x V).
        
        # 1. Calculate S_Row = G @ G.T
        # G is the bipartite adjacency matrix.
        # If G is too large (e.g. Item-Item for 123K items), we cannot materialize S_Col.
        # But we only need S_Col[v, v'] where v and v' share a common neighbor u.
        
        # Let's stick to the definition:
        # Co-occurrence C_Row = G @ G.T
        # Degree D_Row = G.sum(1)
        
        # Jaccard(u1, u2) = C_Row[u1, u2] / (D_Row[u1] + D_Row[u2] - C_Row[u1, u2])
        # SC(u1, u2) = C_Row[u1, u2] / sqrt(D_Row[u1] * D_Row[u2])
        # LHN(u1, u2) = C_Row[u1, u2] / (D_Row[u1] * D_Row[u2])
        # CO(u1, u2) = C_Row[u1, u2]
        
        # Optimization: We don't need the full S matrix if we iterate.
        # But iterating over edges is slow in Python.
        # Let's try to compute S_Row and S_Col densely if possible, or blocked.
        
        # NetEase:
        # U-B: 18K x 22K. S_U: 18Kx18K (OK), S_B: 22Kx22K (OK).
        # U-I: 18K x 123K. S_U: 18Kx18K (OK), S_I: 123Kx123K (Too Big).
        # B-I: 22K x 123K. S_B: 22Kx22K (OK), S_I: 123Kx123K (Too Big).
        
        # So the bottleneck is S_Item (123K).
        # However, we only need to query S_Item for existing edges.
        # For an edge (u, i), we need avg(S_I[i, i']) for all i' interacted by u.
        # i.e., w(u, i) = mean_{i' in N(u)} S_I(i, i')
        #               = mean_{i' in N(u)} ( N(i) intersect N(i') ) / ...
        
        # Let's use the implementation strategy that handles large matrices by batching or sparse.
        # Since we are modifying utility.py, let's implement a robust version.
        
        trend_weight = np.zeros(coo.nnz, dtype=np.float32)
        
        # Convert to CSR for fast slicing
        csr = graph.tocsr()
        csc = graph.tocsc()
        
        # Precompute degrees
        row_deg = np.array(graph.sum(axis=1)).flatten()
        col_deg = np.array(graph.sum(axis=0)).flatten()
        
        # We need to calculate weights for each edge (u, v) in the graph
        # Weight consists of two parts: 
        # 1. User-side collaboration: How similar u is to other users who interacted with v
        # 2. Item-side collaboration: How similar v is to other items interacted by u
        # CAGCN seems to calculate them and put them into the symmetric matrix.
        # For the propagation u -> v, we use User-side collaboration? Or Item-side?
        # CAGCN code: edge_weight[users, i + n_users] = jacard_simi
        # where jacard_simi is mean of similarity between u and other users in N(i).
        # This is "User-based CF" signal injected into the edge (u,i).
        # So for u->i message, we use User-side collaboration (u's similarity to others).
        # For i->u message, we use Item-side collaboration.
        
        # Let's define:
        # W_row[u, v] = Avg_{u' in N(v)} Sim(u, u')
        # W_col[v, u] = Avg_{v' in N(u)} Sim(v, v')
        
        # We will compute these values.
        # Since we have 24G GPU, we can accelerate the Similarity calculation.
        
        # Strategy:
        # 1. Compute Sim Matrix S (Dense or Sparse)
        # 2. For each node, gather neighbors' Sim and average.
        
        # Helper to compute W_row (weights for rows based on Col grouping)
        def compute_side_weights(adj_csr, adj_csc, row_degs, type):
            # adj_csr: [N_row, N_col]
            # We iterate over Cols (v). For each v, get N(v) = {u1, u2...}
            # Calculate Sim(ui, uj) for all pairs in N(v).
            # Assign avg sim to edge (ui, v).
            
            # To vectorize:
            # S = A @ A.T (N_row x N_row).
            # If N_row is small (<30000), compute dense S on GPU.
            # If N_row is large, compute S block-wise or just relevant entries?
            # For U-I (18K users), we can compute 18Kx18K S_user.
            # For I-U (123K items), we CANNOT compute 123Kx123K S_item.
            
            weights = np.zeros(adj_csr.nnz, dtype=np.float32)
            # Map edge (u, v) to index in weights
            # To do this efficiently, let's align with coo structure
            # But coo is not sorted.
            # Let's use a dictionary or just iterate.
            
            # Optimized approach for large scale:
            # Don't precompute full S. Compute on the fly for each col.
            # But that's slow (CAGCN loop).
            
            # Hybrid approach:
            # If dim < 25000, compute full S dense on GPU.
            # If dim >= 25000, compute batch-wise S on GPU.
            
            n_row = adj_csr.shape[0]
            
            # Avoid using torch sparse csr slicing as it's unstable
            # We work with scipy sparse for slicing, then convert to torch dense for computation
            
            batch_size = 2000 # Smaller batch size for safety
            
            # Degrees tensor
            D_row = torch.from_numpy(row_degs).float().to(device).unsqueeze(1) # Nx1
            D_col = torch.from_numpy(col_deg).float().to(device).unsqueeze(0) # 1xM
            
            final_values = []
            final_rows = []
            final_cols = []
            
            for i in range(0, n_row, batch_size):
                end_i = min(i + batch_size, n_row)
                
                # 1. Get block from scipy csr (slicing is fast)
                # A_block_scipy = adj_csr[i:end_i]
                # Convert to dense torch tensor directly
                # If matrix is too big, this dense conversion might be slow/OOM
                # But for batch=2000, 2000 x 123000 float32 is ~1GB.
                # On 24G GPU it is fine. On CPU it might be slow but OK.
                
                A_block_scipy = adj_csr[i:end_i]
                A_block_dense = torch.from_numpy(A_block_scipy.toarray()).float().to(device)
                
                # We need A_torch full for multiplication?
                # S_block = A_block @ A.T
                # A.T is huge.
                # If A is 18K x 123K. A.T is 123K x 18K.
                # A_block (2K x 123K) @ A.T (123K x 18K) -> (2K x 18K). 
                # This is fast and small!
                
                # But we need A_torch on device.
                # If A is too big for dense on device, we keep it sparse?
                # Constructing full sparse A on device might be OK.
                
                # Let's try to keep 'A_full_sparse' on device for multiplication
                # Construct A_full_sparse from indices
                # Using COO for better compatibility
                if i == 0:
                    coo_full = adj_csr.tocoo()
                    indices_full = torch.from_numpy(np.vstack((coo_full.row, coo_full.col)).astype(np.int64)).to(device)
                    values_full = torch.from_numpy(coo_full.data.astype(np.float32)).to(device)
                    shape_full = coo_full.shape
                    A_full_sparse = torch.sparse_coo_tensor(indices_full, values_full, shape_full).to(device)
                
                # S_block = A_block_dense @ A_full_sparse.t()
                # Dense @ Sparse -> Dense
                S_block = torch.matmul(A_block_dense, A_full_sparse.t())
                
                # Apply Sim formula to S_block
                # S_block[u, u'] is count of common neighbors
                C = S_block
                
                if type == 'jc':
                    D1 = D_row[i:end_i]
                    D2 = D_row.t()
                    Union = D1 + D2 - C
                    S_block = C / (Union + 1e-8)
                elif type == 'sc':
                    D1 = D_row[i:end_i]
                    D2 = D_row.t()
                    Mult = torch.sqrt(D1 * D2)
                    S_block = C / (Mult + 1e-8)
                elif type == 'lhn':
                    D1 = D_row[i:end_i]
                    D2 = D_row.t()
                    Mult = D1 * D2
                    S_block = C / (Mult + 1e-8)
                elif type == 'co':
                    pass 
                
                # 2. Compute W_block = S_block @ A_full_sparse
                # (2K x 18K) @ (18K x 123K) -> (2K x 123K).
                # This result is DENSE and potentially large (1GB).
                # We only need values where A_block is non-zero.
                
                # Optimization: 
                # Instead of full matmul, we only care about A_block's edges.
                # W_block_masked = (S_block @ A_full_sparse) * A_block_mask
                
                # But S_block @ A_sparse is efficiently computed?
                # PyTorch: dense @ sparse -> dense.
                # We calculate full W_block, then mask it.
                
                W_block = torch.matmul(S_block, A_full_sparse.to_dense() if A_full_sparse.is_sparse else A_full_sparse)
                # Wait, A_full_sparse.to_dense() is 18K x 123K ~ 9GB. Might OOM.
                # We should use sparse mm if possible.
                # torch.sparse.mm(sparse, dense) -> dense.
                # torch.mm(dense, dense) -> dense.
                # torch.matmul(dense, sparse) -> dense? No, not supported in older torch versions?
                # Let's check.
                
                # If matmul(dense, sparse) is not supported or slow, we can use:
                # W_block = (A_full_sparse.t() @ S_block.t()).t()
                # Sparse.t() @ Dense.t() -> Dense.
                
                W_block = torch.sparse.mm(A_full_sparse.t(), S_block.t()).t()
                
                # 3. Filter: keep only edges where A_block != 0
                # A_block_dense is 1.0 where edge exists.
                W_filtered = W_block * A_block_dense
                
                # 4. Normalize by D_col
                W_filtered = W_filtered / (D_col + 1e-8)
                
                # Extract non-zeros
                indices = W_filtered.nonzero(as_tuple=False)
                rows = indices[:, 0] + i
                cols = indices[:, 1]
                vals = W_filtered[indices[:, 0], indices[:, 1]]
                
                final_rows.append(rows.cpu())
                final_cols.append(cols.cpu())
                final_values.append(vals.cpu())
                
                del S_block, W_block, A_block_dense, W_filtered
                torch.cuda.empty_cache()

            # Concatenate all
            all_rows = torch.cat(final_rows)
            all_cols = torch.cat(final_cols)
            all_vals = torch.cat(final_values)
            
            return all_rows, all_cols, all_vals
        
        # Calculate Row-side collaboration (U-side for U-I graph)
        r_r, r_c, r_v = compute_side_weights(csr, csc, row_deg, type)
        
        # Calculate Col-side collaboration (I-side for U-I graph)
        # Transpose graph: Row becomes Col
        c_r, c_c, c_v = compute_side_weights(csr.transpose(), csc.transpose(), col_deg, type)
        # Note: c_r are cols in original, c_c are rows in original
        
        # Now we have weights for U->I (from User collaboration) and I->U (from Item collaboration)
        # We need to assemble them into the full bipartite adjacency matrix form:
        # [[0, R_weighted], [R_weighted_T, 0]]
        # MultiCBR expects the full symmetric matrix.
        # R_weighted corresponds to U->I edges weighted by User Sim.
        # R_weighted_T corresponds to I->U edges weighted by Item Sim.
        
        # Construct sparse tensor for the full matrix
        # Upper right block: (rows=r_r, cols=r_c + num_rows, vals=r_v)
        # Lower left block: (rows=c_r + num_rows, cols=c_c, vals=c_v)
        
        # But wait, compute_side_weights(csr.transpose()) returns:
        # rows (original cols), cols (original rows).
        # So c_r is original col index, c_c is original row index.
        # This maps to Lower Left block: Row (c_r + num_rows), Col (c_c).
        
        num_rows = csr.shape[0]
        num_cols = csr.shape[1]
        
        idx_upper_row = r_r
        idx_upper_col = r_c + num_rows
        
        idx_lower_row = c_r + num_rows
        idx_lower_col = c_c
        
        all_rows = torch.cat([idx_upper_row, idx_lower_row])
        all_cols = torch.cat([idx_upper_col, idx_lower_col])
        all_vals = torch.cat([r_v, c_v])
        
        # Create sparse tensor
        size = (num_rows + num_cols, num_rows + num_cols)
        indices = torch.stack([all_rows, all_cols])
        
        # Save as coalesced sparse tensor
        trend_sparse = torch.sparse_coo_tensor(indices, all_vals, size).coalesce()
        
        torch.save(trend_sparse, path)
        print(f'Saved trend to {path}')
        
        return trend_sparse


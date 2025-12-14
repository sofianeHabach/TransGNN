import torch as t
import torch.nn as nn
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from Params import args

class PositionalEncodingOptimized(nn.Module):
    """
    Version ULTRA-OPTIMISÉE avec cache complet des encodages
    """
    
    def __init__(self, embedding_dim, max_path_length=10):
        super(PositionalEncodingOptimized, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_path_length = max_path_length
        
        # MLPs pour chaque type d'encodage
        self.spe_mlp = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        self.de_mlp = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        self.pre_mlp = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        # MLP pour combiner
        self.combination_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Stockage données brutes
        self.degrees = None
        self.pageranks = None
        self.shortest_paths = None
        self.nx_graph = None
        self.shortest_paths_cache = {}
        
        # 🔥 NOUVEAU: Cache des encodages calculés
        self.de_cache = None  # (N, d) - Degree encodings
        self.pre_cache = None  # (N, d) - PageRank encodings
        self.spe_cache = None  # (N, N, d) - Shortest path encodings (si N petit)
        self.use_spe_cache = False
        
    def precompute_encodings(self, adj_matrix):
        """
        Pré-calcule TOUS les encodages ET leurs embeddings MLP
        """
        print("📊 Calcul des encodages positionnels...")
        
        num_nodes = adj_matrix.shape[0]
        
        # 1. Degree Encoding
        print("  - Degrees...")
        self.degrees = self._compute_degrees(adj_matrix)
        print("  - Calcul Degree Embeddings MLP...")
        self._precompute_de_cache()
        
        # 2. PageRank Encoding
        print("  - PageRank...")
        self.pageranks = self._compute_pagerank(adj_matrix)
        print("  - Calcul PageRank Embeddings MLP...")
        self._precompute_pre_cache()
        
        # 3. Shortest Paths
        print(f"  - Shortest paths ({num_nodes} nœuds)...")
        
        if num_nodes <= 15000:
            print("    Calcul matrice complète (ceci peut prendre 1-3 minutes)...")
            self.shortest_paths = self._compute_shortest_paths(adj_matrix)
            self.nx_graph = None
            print(f"    ✓ Matrice {num_nodes}x{num_nodes} calculée")
            
            # 🔥 Pré-calculer TOUS les SPE embeddings si N petit
            if num_nodes <= 10000:
                print("  - Calcul SPE Embeddings MLP (ceci peut prendre 2-3 minutes)...")
                self._precompute_spe_cache()
        else:
            print("    Graphe très grand, création graphe NetworkX pour cache BFS...")
            self.shortest_paths = None
            adj_scipy = self._to_scipy_sparse(adj_matrix)
            self.nx_graph = nx.from_scipy_sparse_array(adj_scipy)
            print(f"    ✓ Graphe NetworkX créé (SPE calculés à la volée)")
        
        print("✅ Encodages positionnels calculés et mis en cache")
    
    def _precompute_de_cache(self):
        """Pré-calcule tous les Degree Embeddings"""
        N = len(self.degrees)
        degrees_tensor = t.tensor(self.degrees, dtype=t.float32).cuda().unsqueeze(1)  # (N, 1)
        
        with t.no_grad():
            self.de_cache = self.de_mlp(degrees_tensor)  # (N, d)
        
        print(f"    ✓ Cache DE: {self.de_cache.shape}")
    
    def _precompute_pre_cache(self):
        """Pré-calcule tous les PageRank Embeddings"""
        N = len(self.pageranks)
        pr_tensor = t.tensor(self.pageranks, dtype=t.float32).cuda().unsqueeze(1)  # (N, 1)
        
        with t.no_grad():
            self.pre_cache = self.pre_mlp(pr_tensor)  # (N, d)
        
        print(f"    ✓ Cache PRE: {self.pre_cache.shape}")
    
    def _precompute_spe_cache(self):
        """
        Pré-calcule TOUS les SPE embeddings
        Attention: Matrice NxNxd peut être énorme!
        """
        N = self.shortest_paths.shape[0]
        d = self.embedding_dim
        
        # Pour économiser mémoire, calculer par batch
        batch_size = 500
        self.spe_cache = t.zeros((N, N, d), dtype=t.float32).cuda()
        
        total_batches = (N + batch_size - 1) // batch_size
        
        with t.no_grad():
            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                batch_i_size = end_i - i
                
                if i % 2000 == 0:
                    print(f"    Batch {i//batch_size + 1}/{total_batches}...")
                
                # Pour ce batch de sources, calculer SPE vers toutes destinations
                for j in range(0, N, batch_size):
                    end_j = min(j + batch_size, N)
                    
                    # Distances pour ce sous-bloc (batch_i_size, batch_j_size)
                    dist_block = t.tensor(
                        self.shortest_paths[i:end_i, j:end_j],
                        dtype=t.float32
                    ).cuda()
                    
                    # Aplatir et passer par MLP
                    flat_dist = dist_block.view(-1, 1)  # (batch_i_size * batch_j_size, 1)
                    spe_embeds = self.spe_mlp(flat_dist)  # (batch_i_size * batch_j_size, d)
                    
                    # Reshape et stocker
                    self.spe_cache[i:end_i, j:end_j] = spe_embeds.view(
                        batch_i_size, end_j - j, d
                    )
        
        self.use_spe_cache = True
        print(f"    ✓ Cache SPE: {self.spe_cache.shape} ({self.spe_cache.numel() * 4 / 1e9:.2f} GB)")
    
    def _compute_degrees(self, adj_matrix):
        """Calcule le degré de chaque nœud"""
        if isinstance(adj_matrix, t.Tensor):
            if adj_matrix.is_sparse:
                degrees = t.sparse.sum(adj_matrix, dim=1).to_dense().cpu().numpy()
            else:
                degrees = adj_matrix.sum(dim=1).cpu().numpy()
        else:
            degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        
        return degrees
    
    def _compute_pagerank(self, adj_matrix):
        """Calcule le PageRank"""
        adj_scipy = self._to_scipy_sparse(adj_matrix)
        G = nx.from_scipy_sparse_array(adj_scipy)
        pr = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-4)
        pageranks = np.array([pr.get(i, 0) for i in range(adj_scipy.shape[0])])
        return pageranks
    
    def _compute_shortest_paths(self, adj_matrix):
        """Calcule matrice complète des plus courts chemins"""
        adj_scipy = self._to_scipy_sparse(adj_matrix)
        dist_matrix = shortest_path(
            adj_scipy, 
            directed=False, 
            unweighted=True,
            method='auto'
        )
        dist_matrix[np.isinf(dist_matrix)] = self.max_path_length
        return dist_matrix
    
    def _compute_shortest_path_bfs(self, source, target):
        """Calcule plus court chemin entre 2 nœuds via BFS"""
        cache_key = (source, target)
        if cache_key in self.shortest_paths_cache:
            return self.shortest_paths_cache[cache_key]
        
        try:
            path_length = nx.shortest_path_length(self.nx_graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_length = self.max_path_length
        
        self.shortest_paths_cache[cache_key] = path_length
        
        if len(self.shortest_paths_cache) > 50000:
            keys_to_remove = list(self.shortest_paths_cache.keys())[:10000]
            for k in keys_to_remove:
                del self.shortest_paths_cache[k]
        
        return path_length
    
    def _to_scipy_sparse(self, adj_matrix):
        """Convertit en scipy sparse"""
        if isinstance(adj_matrix, t.Tensor):
            if adj_matrix.is_sparse:
                indices = adj_matrix._indices().cpu().numpy()
                values = adj_matrix._values().cpu().numpy()
                shape = adj_matrix.shape
                return csr_matrix(
                    (values, (indices[0], indices[1])), 
                    shape=shape
                )
            else:
                return csr_matrix(adj_matrix.cpu().numpy())
        return adj_matrix
    
    def get_spe(self, node_i, node_j):
        """Shortest Path Encoding - version avec cache"""
        # 🔥 Si cache complet disponible
        if self.use_spe_cache:
            return self.spe_cache[node_i, node_j]
        
        # Sinon calculer
        if self.shortest_paths is not None:
            distance = self.shortest_paths[node_i, node_j]
        else:
            distance = self._compute_shortest_path_bfs(node_i, node_j)
        
        distance_tensor = t.tensor([[distance]], dtype=t.float32).cuda()
        spe = self.spe_mlp(distance_tensor).squeeze(0)
        return spe
    
    def get_de(self, node_id):
        """Degree Encoding - version avec cache"""
        return self.de_cache[node_id]
    
    def get_pre(self, node_id):
        """PageRank Encoding - version avec cache"""
        return self.pre_cache[node_id]
    
    def forward(self, node_i, sample_nodes, raw_embeddings):
        """
        Combine raw embeddings avec positional encodings
        Version OPTIMISÉE avec cache
        """
        enriched = []
        
        # 1. Nœud central
        x_i = raw_embeddings[node_i]
        spe_i = self.get_spe(node_i, node_i)
        de_i = self.get_de(node_i)
        pre_i = self.get_pre(node_i)
        
        concat_i = t.cat([x_i, spe_i, de_i, pre_i], dim=0)
        h_i = self.combination_mlp(concat_i)
        enriched.append(h_i)
        
        # 2. Nœuds échantillons
        for j in sample_nodes:
            j = j.item()
            x_j = raw_embeddings[j]
            spe_j = self.get_spe(node_i, j)
            de_j = self.get_de(j)
            pre_j = self.get_pre(j)
            
            concat_j = t.cat([x_j, spe_j, de_j, pre_j], dim=0)
            h_j = self.combination_mlp(concat_j)
            enriched.append(h_j)
        
        return t.stack(enriched, dim=0)


def create_positional_encoding(handler, embedding_dim):
    """Crée le module de positional encoding optimisé"""
    print("\n🔧 Création du module Positional Encoding...")
    
    pos_encoder = PositionalEncodingOptimized(
        embedding_dim=embedding_dim,
        max_path_length=10
    ).cuda()
    
    # Pré-calculer tous les encodages
    pos_encoder.precompute_encodings(handler.torchBiAdj)
    
    return pos_encoder
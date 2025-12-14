import torch as t
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from Params import args

class AttentionSampling:
    """
    Module d'échantillonnage d'attention (Section 3.2 de l'article)
    Version corrigée et optimisée
    """
    
    def __init__(self, num_nodes, embedding_dim, sample_size=20, alpha=0.5):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.sample_size = sample_size
        self.alpha = alpha
        self.attention_samples = None
        self.attention_scores = None
        
    def compute_similarity_matrix(self, embeddings, adj_matrix=None):
        """
        Calcule la matrice de similarité selon Eq. 1 et 2 de l'article
        Version optimisée pour éviter out-of-memory
        """
        # Étape 1: Similarité sémantique brute (Eq. 1)
        embeddings_normalized = t.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Pour grands graphes, calculer par batch
        if self.num_nodes > 10000:
            similarity = self._compute_similarity_batched(embeddings_normalized)
        else:
            similarity = t.mm(embeddings_normalized, embeddings_normalized.t())
        
        # Étape 2: Incorporer préférences des voisins (Eq. 2)
        if adj_matrix is not None and self.alpha > 0:
            # S = S + α * Â * S
            # Convertir S en dense si sparse (pour calcul efficace)
            if similarity.is_sparse:
                similarity_dense = similarity.to_dense()
            else:
                similarity_dense = similarity
            
            # Multiplication sparse @ dense
            similarity_propagated = t.sparse.mm(adj_matrix, similarity_dense)
            
            # Ajouter identité à adj_matrix (self-loops)
            # Créer matrice identité sparse
            indices = t.arange(self.num_nodes).unsqueeze(0).repeat(2, 1).cuda()
            values = t.ones(self.num_nodes).cuda()
            identity = t.sparse_coo_tensor(indices, values, adj_matrix.shape).cuda()
            
            # Â = A + I
            adj_with_self = adj_matrix + identity
            
            # Propager similarité
            similarity_propagated = t.sparse.mm(adj_with_self, similarity_dense)
            similarity = similarity_dense + self.alpha * similarity_propagated
        
        return similarity
    
    def _compute_similarity_batched(self, embeddings_normalized, batch_size=1000):
        """
        Calcule similarité par batch pour économiser mémoire
        """
        N = embeddings_normalized.shape[0]
        similarity = t.zeros((N, N), device=embeddings_normalized.device)
        
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_i = embeddings_normalized[i:end_i]
            
            # Calculer similarité pour ce batch
            sim_batch = t.mm(batch_i, embeddings_normalized.t())
            similarity[i:end_i] = sim_batch
        
        return similarity
    
    def sample_attention_nodes(self, similarity_matrix, exclude_self=True):
        """
        Échantillonne les top-k nœuds les plus similaires
        Version optimisée par batch
        """
        N = similarity_matrix.shape[0]
        k = min(self.sample_size, N - 1) if exclude_self else self.sample_size
        
        if exclude_self:
            # Mettre -inf sur la diagonale
            mask = t.eye(N, dtype=t.bool, device=similarity_matrix.device)
            similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Échantillonner par batch pour économiser mémoire
        batch_size = 2000
        all_indices = []
        all_scores = []
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_sim = similarity_matrix[start:end]
            
            top_scores, top_indices = t.topk(batch_sim, k, dim=1)
            
            # Remplacer -inf par 0
            top_scores = t.where(
                t.isinf(top_scores), 
                t.zeros_like(top_scores), 
                top_scores
            )
            
            all_indices.append(top_indices)
            all_scores.append(top_scores)
        
        self.attention_samples = t.cat(all_indices, dim=0)
        self.attention_scores = t.cat(all_scores, dim=0)
        
        return self.attention_samples, self.attention_scores
    
    def get_attention_samples(self, node_ids):
        """Récupère échantillons pour nœuds spécifiques"""
        if self.attention_samples is None:
            raise ValueError("Appelez d'abord sample_attention_nodes()")
        
        return self.attention_samples[node_ids]
    
    def get_sampled_embeddings(self, embeddings, node_ids):
        """Récupère embeddings des échantillons"""
        if isinstance(node_ids, int):
            node_ids = t.tensor([node_ids], device=embeddings.device)
            squeeze_output = True
        else:
            squeeze_output = False
        
        sample_indices = self.get_attention_samples(node_ids)
        sampled_embeds = embeddings[sample_indices]
        
        if squeeze_output:
            sampled_embeds = sampled_embeds.squeeze(0)
        
        return sampled_embeds
    
    def update_samples(self, embeddings, adj_matrix=None):
        """Recalcule les échantillons avec nouveaux embeddings"""
        print("   Recalcul similarité...")
        similarity = self.compute_similarity_matrix(embeddings, adj_matrix)
        print("   Échantillonnage top-k...")
        self.sample_attention_nodes(similarity)
        print("   ✓ Échantillons mis à jour")


def create_attention_sampling(handler, sample_size=20):
    """
    Crée et initialise le module d'attention sampling
    Version simplifiée pour éviter erreurs
    """
    num_nodes = args.user + args.item
    embedding_dim = args.latdim
    
    sampler = AttentionSampling(
        num_nodes=num_nodes,
        embedding_dim=embedding_dim,
        sample_size=sample_size,
        alpha=0.5
    )
    
    print("🔍 Initialisation de l'attention sampling...")
    
    # Initialiser avec embeddings aléatoires
    initial_embeds = t.randn(num_nodes, embedding_dim).cuda()
    
    # Calculer similarité SANS adjacence pour la première fois
    # (évite les problèmes avec matrice sparse)
    print("   Calcul similarité initiale (sémantique seulement)...")
    similarity = sampler.compute_similarity_matrix(
        initial_embeds, 
        adj_matrix=None  # Pas d'adjacence pour l'init
    )
    
    print("   Échantillonnage des top-k nœuds...")
    sampler.sample_attention_nodes(similarity)
    
    print(f"✅ Échantillonnage créé: {sample_size} nœuds par nœud central")
    
    return sampler
import torch as t
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from Params import args

class AttentionSampling:
    """
    Module d'échantillonnage d'attention (Section 3.2 de l'article)
    Version MEMORY-EFFICIENT pour grands graphes
    """
    
    def __init__(self, num_nodes, embedding_dim, sample_size=20, alpha=0.5):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.sample_size = sample_size
        self.alpha = alpha
        self.attention_samples = None
        self.attention_scores = None
        
    def compute_similarity_batched(self, embeddings, batch_size=1000):
        """
        Calcule top-k similarités SANS créer la matrice complète NxN
        Économie massive de mémoire!
        
        Args:
            embeddings: (N, d)
            batch_size: Taille des batches
            
        Returns:
            top_indices: (N, k) - indices des k plus similaires
            top_scores: (N, k) - scores de similarité
        """
        N = embeddings.shape[0]
        k = min(self.sample_size, N - 1)
        
        # Normaliser embeddings
        embeddings_normalized = t.nn.functional.normalize(embeddings, p=2, dim=1)
        
        all_indices = []
        all_scores = []
        
        print(f"   Échantillonnage par batch ({batch_size} nœuds/batch)...")
        total_batches = (N + batch_size - 1) // batch_size
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            
            if start % 10000 == 0:
                print(f"     Batch {start//batch_size + 1}/{total_batches}...")
            
            # Batch de queries (batch_size, d)
            batch_queries = embeddings_normalized[start:end]
            
            # Similarité avec TOUS les nœuds (batch_size, N)
            batch_similarities = t.mm(batch_queries, embeddings_normalized.t())
            
            # Masquer self-loops (mettre -inf sur diagonale)
            for i in range(batch_similarities.shape[0]):
                global_idx = start + i
                batch_similarities[i, global_idx] = float('-inf')
            
            # Top-k pour ce batch
            top_scores, top_indices = t.topk(batch_similarities, k, dim=1)
            
            # Remplacer -inf par 0
            top_scores = t.where(
                t.isinf(top_scores),
                t.zeros_like(top_scores),
                top_scores
            )
            
            all_indices.append(top_indices)
            all_scores.append(top_scores)
            
            # Libérer mémoire
            del batch_similarities
            t.cuda.empty_cache()
        
        # Concatener résultats
        self.attention_samples = t.cat(all_indices, dim=0)  # (N, k)
        self.attention_scores = t.cat(all_scores, dim=0)    # (N, k)
        
        return self.attention_samples, self.attention_scores
    
    def compute_similarity_with_propagation(self, embeddings, adj_matrix, batch_size=1000):
        """
        Version avec propagation de voisinage (Eq. 2 de l'article)
        Mais en mode memory-efficient
        
        S = S_semantic + α * A * S_semantic
        """
        N = embeddings.shape[0]
        k = min(self.sample_size, N - 1)
        
        # Normaliser embeddings
        embeddings_normalized = t.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Propager via adjacence (A * E)
        if adj_matrix is not None and self.alpha > 0:
            print("   Propagation via matrice d'adjacence...")
            propagated_embeds = t.sparse.mm(adj_matrix, embeddings_normalized)
            
            # Combiner: E_final = E + α * (A * E)
            embeddings_combined = embeddings_normalized + self.alpha * propagated_embeds
            embeddings_combined = t.nn.functional.normalize(embeddings_combined, p=2, dim=1)
        else:
            embeddings_combined = embeddings_normalized
        
        # Échantillonner sur embeddings combinés
        return self.compute_similarity_batched(embeddings_combined, batch_size)
    
    def sample_attention_nodes(self, embeddings, adj_matrix=None, use_propagation=False):
        """
        Échantillonne les top-k nœuds les plus similaires
        Version memory-efficient
        
        Args:
            embeddings: (N, d) - peut être tensor ou on calcule depuis scratch
            adj_matrix: Matrice d'adjacence sparse (optionnel)
            use_propagation: Si True, utilise Eq. 2 avec propagation de voisinage
        """
        # Déterminer batch size basé sur mémoire disponible
        if self.num_nodes > 50000:
            batch_size = 500  # Très grand graphe
        elif self.num_nodes > 20000:
            batch_size = 1000  # Grand graphe
        else:
            batch_size = 2000  # Graphe moyen
        
        if use_propagation and adj_matrix is not None:
            self.compute_similarity_with_propagation(embeddings, adj_matrix, batch_size)
        else:
            self.compute_similarity_batched(embeddings, batch_size)
        
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
    
    def update_samples(self, embeddings, adj_matrix=None, use_propagation=False):
        """Recalcule les échantillons avec nouveaux embeddings"""
        print("   🔄 Recalcul similarité...")
        self.sample_attention_nodes(embeddings, adj_matrix, use_propagation)
        print("   ✓ Échantillons mis à jour")


def create_attention_sampling(handler, sample_size=20, use_propagation=False):
    """
    Crée et initialise le module d'attention sampling
    Version memory-efficient pour grands graphes
    
    Args:
        handler: DataHandler
        sample_size: Nombre d'échantillons (k)
        use_propagation: Si True, utilise Eq. 2 avec propagation
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
    print(f"   Graphe: {num_nodes} nœuds ({args.user} users + {args.item} items)")
    
    # Initialiser avec embeddings aléatoires
    initial_embeds = t.randn(num_nodes, embedding_dim).cuda()
    
    # Calculer similarité et échantillonner
    print("   Calcul similarité initiale...")
    sampler.sample_attention_nodes(
        initial_embeds,
        adj_matrix=handler.torchBiAdj if use_propagation else None,
        use_propagation=use_propagation
    )
    
    print(f"✅ Échantillonnage créé: {sample_size} nœuds par nœud central")
    
    # Estimer mémoire utilisée
    memory_mb = (num_nodes * sample_size * 4) / (1024**2)  # 4 bytes par int32
    print(f"   Mémoire cache: ~{memory_mb:.1f} MB")
    
    return sampler
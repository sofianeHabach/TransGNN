import torch as t
import torch.nn as nn
import numpy as np
from Params import args

class SampleUpdater:
    """
    Module de mise à jour des échantillons d'attention (Section 3.4.3)
    Implémente 2 stratégies: Message Passing et Random Walk
    """
    
    def __init__(self, attention_sampler, update_strategy='message_passing'):
        """
        Args:
            attention_sampler: Instance d'AttentionSampling
            update_strategy: 'message_passing' ou 'random_walk'
        """
        self.attention_sampler = attention_sampler
        self.update_strategy = update_strategy
        
    def update_via_message_passing(self, adj_matrix, embeddings):
        """
        Stratégie 1: Message Passing Update (Eq. 12 de l'article)
        
        Principe: Pour chaque nœud, agrège les échantillons d'attention
        de ses voisins pour découvrir de nouveaux nœuds pertinents
        
        Attn_Msg(v_i) = ∪ Smp(v_j) ∀v_j ∈ N(v_i)
        
        Args:
            adj_matrix: Matrice d'adjacence sparse
            embeddings: Embeddings actuels (N, d)
        """
        N = embeddings.shape[0]
        k = self.attention_sampler.sample_size
        
        # Récupérer les échantillons actuels
        current_samples = self.attention_sampler.attention_samples  # (N, k)
        
        # Pour chaque nœud, collecter les échantillons de ses voisins
        new_candidates = []
        
        if adj_matrix.is_sparse:
            # Convertir matrice sparse en format COO pour itération
            indices = adj_matrix._indices()
            
            # Créer dictionnaire voisins pour chaque nœud
            neighbors_dict = {}
            for i in range(indices.shape[1]):
                src = indices[0, i].item()
                dst = indices[1, i].item()
                if src not in neighbors_dict:
                    neighbors_dict[src] = []
                neighbors_dict[src].append(dst)
        
        for node_i in range(N):
            # Collecter échantillons des voisins: Attn_Msg(v_i)
            if node_i in neighbors_dict:
                neighbor_samples = []
                for neighbor in neighbors_dict[node_i]:
                    neighbor_samples.extend(
                        current_samples[neighbor].cpu().numpy().tolist()
                    )
                
                # Ajouter aussi les échantillons actuels du nœud
                neighbor_samples.extend(
                    current_samples[node_i].cpu().numpy().tolist()
                )
                
                # Enlever doublons
                neighbor_samples = list(set(neighbor_samples))
            else:
                # Si pas de voisins, garder échantillons actuels
                neighbor_samples = current_samples[node_i].cpu().numpy().tolist()
            
            new_candidates.append(neighbor_samples)
        
        # Re-sélectionner top-k parmi les candidats basés sur similarité
        updated_samples = t.zeros((N, k), dtype=t.long).cuda()
        
        embeddings_normalized = t.nn.functional.normalize(embeddings, p=2, dim=1)
        
        for node_i in range(N):
            candidates = new_candidates[node_i]
            
            if len(candidates) <= k:
                # Pas assez de candidats, garder tous + padding
                padded = candidates + [node_i] * (k - len(candidates))
                updated_samples[node_i] = t.tensor(padded[:k]).cuda()
            else:
                # Calculer similarité avec candidats
                candidate_embeds = embeddings_normalized[candidates]
                query_embed = embeddings_normalized[node_i].unsqueeze(0)
                
                similarities = t.mm(query_embed, candidate_embeds.t()).squeeze(0)
                
                # Sélectionner top-k
                top_k_indices = t.topk(similarities, k).indices
                selected = [candidates[idx] for idx in top_k_indices.cpu().numpy()]
                updated_samples[node_i] = t.tensor(selected).cuda()
        
        # Mettre à jour l'attention sampler
        self.attention_sampler.attention_samples = updated_samples
    
    def update_via_random_walk(self, adj_matrix, embeddings, walk_length=5):
        """
        Stratégie 2: Random Walk Update (Eq. 11 de l'article)
        
        Principe: Pour chaque échantillon actuel, faire une marche aléatoire
        guidée par la similarité pour découvrir des nœuds pertinents proches
        
        p_i→j = (h_i * h_j^T) / Σ(h_i * h_l^T) si v_j ∈ N(v_i)
        
        Args:
            adj_matrix: Matrice d'adjacence sparse
            embeddings: Embeddings actuels (N, d)
            walk_length: Longueur de la marche (L dans l'article)
        """
        N = embeddings.shape[0]
        k = self.attention_sampler.sample_size
        current_samples = self.attention_sampler.attention_samples  # (N, k)
        
        # Normaliser embeddings pour calcul similarité
        embeddings_normalized = t.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Construire dictionnaire des voisins
        if adj_matrix.is_sparse:
            indices = adj_matrix._indices()
            neighbors_dict = {}
            for i in range(indices.shape[1]):
                src = indices[0, i].item()
                dst = indices[1, i].item()
                if src not in neighbors_dict:
                    neighbors_dict[src] = []
                neighbors_dict[src].append(dst)
        
        # Pour chaque nœud
        updated_samples = t.zeros((N, k), dtype=t.long).cuda()
        
        for node_i in range(N):
            # Collecter candidats via random walks depuis chaque échantillon
            all_explored = set()
            
            for sample_node in current_samples[node_i].cpu().numpy():
                # Random walk depuis ce nœud échantillon
                current = sample_node
                
                for step in range(walk_length):
                    all_explored.add(current)
                    
                    # Voisins du nœud actuel
                    if current not in neighbors_dict or len(neighbors_dict[current]) == 0:
                        break
                    
                    neighbors = neighbors_dict[current]
                    
                    # Calculer probabilités de transition (Eq. 11)
                    current_embed = embeddings_normalized[current]
                    neighbor_embeds = embeddings_normalized[neighbors]
                    
                    # Similarités avec voisins
                    similarities = t.mm(
                        current_embed.unsqueeze(0), 
                        neighbor_embeds.t()
                    ).squeeze(0)
                    
                    # Normaliser en probabilités
                    probs = t.softmax(similarities, dim=0)
                    
                    # Échantillonner prochain nœud
                    next_idx = t.multinomial(probs, 1).item()
                    current = neighbors[next_idx]
            
            # Convertir en liste
            explored_list = list(all_explored)
            
            # Sélectionner top-k parmi nœuds explorés basé sur similarité
            if len(explored_list) <= k:
                padded = explored_list + [node_i] * (k - len(explored_list))
                updated_samples[node_i] = t.tensor(padded[:k]).cuda()
            else:
                query_embed = embeddings_normalized[node_i].unsqueeze(0)
                explored_embeds = embeddings_normalized[explored_list]
                
                similarities = t.mm(query_embed, explored_embeds.t()).squeeze(0)
                top_k_indices = t.topk(similarities, k).indices
                
                selected = [explored_list[idx] for idx in top_k_indices.cpu().numpy()]
                updated_samples[node_i] = t.tensor(selected).cuda()
        
        # Mettre à jour
        self.attention_sampler.attention_samples = updated_samples
    
    def update(self, adj_matrix, embeddings, **kwargs):
        """
        Met à jour les échantillons selon la stratégie choisie
        
        Args:
            adj_matrix: Matrice d'adjacence
            embeddings: Embeddings actuels
            **kwargs: Arguments additionnels (walk_length, etc.)
        """
        if self.update_strategy == 'message_passing':
            self.update_via_message_passing(adj_matrix, embeddings)
        elif self.update_strategy == 'random_walk':
            walk_length = kwargs.get('walk_length', 5)
            self.update_via_random_walk(adj_matrix, embeddings, walk_length)
        else:
            raise ValueError(f"Stratégie inconnue: {self.update_strategy}")


class AdaptiveSampleUpdater:
    """
    Version optimisée qui décide quand mettre à jour les échantillons
    """
    
    def __init__(self, sample_updater, update_frequency=3):
        """
        Args:
            sample_updater: Instance de SampleUpdater
            update_frequency: Mettre à jour tous les N epochs
        """
        self.sample_updater = sample_updater
        self.update_frequency = update_frequency
        self.current_epoch = 0
    
    def should_update(self):
        """Décide si on doit mettre à jour ce epoch"""
        return self.current_epoch % self.update_frequency == 0
    
    def step(self, adj_matrix, embeddings):
        """
        Appelé à chaque epoch
        
        Args:
            adj_matrix: Matrice d'adjacence
            embeddings: Embeddings actuels
        """
        if self.should_update():
            print(f"🔄 Mise à jour des échantillons (epoch {self.current_epoch})...")
            self.sample_updater.update(adj_matrix, embeddings)
        
        self.current_epoch += 1


def create_sample_updater(attention_sampler, strategy='message_passing', 
                         update_frequency=3):
    """
    Crée le module de mise à jour des échantillons
    
    Args:
        attention_sampler: Instance d'AttentionSampling
        strategy: 'message_passing' (recommandé) ou 'random_walk'
        update_frequency: Fréquence de mise à jour (en epochs)
        
    Returns:
        updater: Instance d'AdaptiveSampleUpdater
    """
    base_updater = SampleUpdater(attention_sampler, strategy)
    adaptive_updater = AdaptiveSampleUpdater(base_updater, update_frequency)
    
    print(f"✅ Sample updater créé (stratégie: {strategy}, "
          f"fréquence: tous les {update_frequency} epochs)")
    
    return adaptive_updater

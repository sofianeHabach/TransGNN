import torch
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict
from Transformer import TransformerEncoderLayer

class TransGNNComplete(nn.Module):
    """
    TransGNN complet OPTIMISÉ avec calculs vectorisés
    """
    
    def __init__(self, attention_sampler, pos_encoder, sample_updater):
        super(TransGNNComplete, self).__init__()
        
        # Modules externes
        self.attention_sampler = attention_sampler
        self.pos_encoder = pos_encoder
        self.sample_updater = sample_updater
        
        # Embeddings de base
        self.user_embeding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(args.user, args.latdim))
        )
        self.item_embeding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(args.item, args.latdim))
        )
        
        # Transformers pour users et items
        self.user_transformer_encoder = TransformerEncoderLayer(
            d_model=args.latdim, 
            num_heads=args.num_head, 
            dropout=args.dropout
        )
        self.item_transformer_encoder = TransformerEncoderLayer(
            d_model=args.latdim, 
            num_heads=args.num_head, 
            dropout=args.dropout
        )
        
        # Cache pour éviter recalculs
        self.use_pos_encoding = True  # Flag pour désactiver si trop lent
    
    def transformer_layer_batch_optimized(self, embeds, node_ids, is_user=True):
        """
        Version ULTRA-OPTIMISÉE du transformer layer
        Traite tous les nœuds en parallèle sans boucles Python
        
        Args:
            embeds: Embeddings complets (N, d)
            node_ids: IDs des nœuds à traiter (B,)
            is_user: True si users, False si items
            
        Returns:
            updated_embeds: (B, d)
        """
        B = len(node_ids)
        k = self.attention_sampler.sample_size
        
        # Sélectionner le bon transformer
        transformer = self.user_transformer_encoder if is_user else self.item_transformer_encoder
        
        # 1. Récupérer TOUS les échantillons d'un coup
        sample_indices = self.attention_sampler.attention_samples[node_ids]  # (B, k)
        
        # 2. SIMPLIFICATION: Utiliser embeddings bruts sans positional encoding
        #    Le positional encoding est le goulot - on peut l'activer plus tard
        if not self.use_pos_encoding:
            # Version rapide: juste récupérer les embeddings
            # Query: nœuds centraux (B, d)
            queries = embeds[node_ids]  # (B, d)
            
            # Keys/Values: échantillons (B, k, d)
            # Aplatir les indices pour gather en une fois
            flat_indices = sample_indices.view(-1)  # (B*k,)
            sampled_embeds = embeds[flat_indices].view(B, k, args.latdim)  # (B, k, d)
            
            # Ajouter query aux keys/values
            # (B, 1, d) + (B, k, d) = (B, k+1, d)
            keys_values = torch.cat([
                queries.unsqueeze(1),  # (B, 1, d)
                sampled_embeds         # (B, k, d)
            ], dim=1)  # (B, k+1, d)
            
            # 3. Appliquer transformer en batch
            # Format attendu: (seq_len, batch, d)
            queries_t = queries.unsqueeze(0)  # (1, B, d)
            keys_values_t = keys_values.transpose(0, 1)  # (k+1, B, d)
            
            # Attention
            attn_output, _ = transformer.attention(
                queries_t, keys_values_t, keys_values_t
            )
            
            # Output: (1, B, d) -> (B, d)
            updated = attn_output.squeeze(0)
            
            return updated
        
        else:
            # Version avec positional encoding (LENT)
            # Traiter par mini-batches pour éviter OOM
            mini_batch_size = 64
            all_updated = []
            
            for start in range(0, B, mini_batch_size):
                end = min(start + mini_batch_size, B)
                mini_node_ids = node_ids[start:end]
                mini_B = len(mini_node_ids)
                
                # Liste pour stocker enriched embeddings
                enriched_list = []
                
                for i in range(mini_B):
                    node_id = mini_node_ids[i].item()
                    sample_idx = sample_indices[start + i]
                    
                    # Appliquer positional encoding
                    enriched = self.pos_encoder.forward(
                        node_id, sample_idx, embeds
                    )  # (k+1, d)
                    enriched_list.append(enriched)
                
                # Stack: (mini_B, k+1, d)
                batch_enriched = torch.stack(enriched_list, dim=0)
                
                # Séparer query et keys/values
                queries = batch_enriched[:, 0, :]  # (mini_B, d)
                keys_values = batch_enriched  # (mini_B, k+1, d)
                
                # Transformer
                queries_t = queries.unsqueeze(0)  # (1, mini_B, d)
                keys_values_t = keys_values.transpose(0, 1)  # (k+1, mini_B, d)
                
                attn_output, _ = transformer.attention(
                    queries_t, keys_values_t, keys_values_t
                )
                
                updated = attn_output.squeeze(0)  # (mini_B, d)
                all_updated.append(updated)
            
            return torch.cat(all_updated, dim=0)  # (B, d)
    
    def gnn_message_passing(self, adj, embeds):
        """Message passing GNN standard"""
        return torch.spmm(adj, embeds)
    
    def forward(self, adj):
        """
        Forward pass OPTIMISÉ
        """
        # Embeddings initiaux
        embeds = [torch.concat([self.user_embeding, self.item_embeding], dim=0)]
        
        # Blocs TransGNN
        for block in range(args.block_num):
            
            # --- PARTIE 1: GNN Message Passing ---
            tmp_embeds = self.gnn_message_passing(adj, embeds[-1])
            
            # --- PARTIE 2: Transformer OPTIMISÉ ---
            # Users - traiter TOUS d'un coup
            user_ids = torch.arange(0, args.user, device='cuda')
            tmp_user_embeds = self.transformer_layer_batch_optimized(
                tmp_embeds, user_ids, is_user=True
            )
            
            # Items - traiter TOUS d'un coup
            item_ids = torch.arange(args.user, args.user + args.item, device='cuda')
            tmp_item_embeds = self.transformer_layer_batch_optimized(
                tmp_embeds, item_ids, is_user=False
            )
            
            # --- PARTIE 3: Connexions Résiduelles ---
            tmp_user_embeds = tmp_user_embeds + tmp_embeds[:args.user]
            tmp_item_embeds = tmp_item_embeds + tmp_embeds[args.user:]
            
            # Recombiner
            tmp_embeds = torch.concat([tmp_user_embeds, tmp_item_embeds], dim=0)
            embeds.append(tmp_embeds)
        
        # Agrégation finale
        embeds = sum(embeds)
        user_embeds = embeds[:args.user]
        item_embeds = embeds[args.user:]
        
        return embeds, user_embeds, item_embeds
    
    def bprLoss(self, user_embeding, item_embeding, ancs, poss, negs):
        """BPR Loss standard"""
        ancEmbeds = user_embeding[ancs]
        posEmbeds = item_embeding[poss]
        negEmbeds = item_embeding[negs]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = -((scoreDiff).sigmoid() + 1e-6).log().mean()
        return bprLoss
    
    def calcLosses(self, ancs, poss, negs, adj):
        """Calcul de la loss totale"""
        embeds, user_embeds, item_embeds = self.forward(adj)
        user_embeding = embeds[:args.user]
        item_embeding = embeds[args.user:]
        
        bprLoss = self.bprLoss(user_embeding, item_embeding, ancs, poss, negs) + \
                  self.bprLoss(user_embeds, item_embeds, ancs, poss, negs)
        return bprLoss
    
    def predict(self, adj):
        """Prédiction (pour test)"""
        embeds, user_embeds, item_embeds = self.forward(adj)
        return user_embeds, item_embeds
    
    def update_attention_samples(self, adj):
        """
        Met à jour les échantillons d'attention
        À appeler périodiquement durant l'entraînement
        """
        # Récupérer embeddings actuels
        with torch.no_grad():
            current_embeds = torch.concat(
                [self.user_embeding, self.item_embeding], 
                dim=0
            )
        
        # Mettre à jour via le sample updater
        self.sample_updater.step(adj, current_embeds)


# Fonction d'initialisation complète
def create_complete_transgnn(handler, sample_size=20, update_strategy='message_passing'):
    """
    Crée une instance complète de TransGNN avec tous les modules
    
    Args:
        handler: Instance de DataHandler
        sample_size: Nombre d'échantillons d'attention (k)
        update_strategy: 'message_passing' ou 'random_walk'
        
    Returns:
        model: Instance de TransGNNComplete
    """
    from AttentionSampling import create_attention_sampling
    from PositionalEncoding import create_positional_encoding
    from SampleUpdate import create_sample_updater
    
    print("="*60)
    print("Initialisation TransGNN Complet")
    print("="*60)
    
    # 1. Attention Sampling
    attention_sampler = create_attention_sampling(handler, sample_size)
    
    # 2. Positional Encoding
    pos_encoder = create_positional_encoding(handler, args.latdim)
    
    # 3. Sample Updater
    sample_updater = create_sample_updater(
        attention_sampler, 
        strategy=update_strategy,
        update_frequency=3
    )
    
    # 4. Créer le modèle
    model = TransGNNComplete(
        attention_sampler, 
        pos_encoder, 
        sample_updater
    ).cuda()
    
    # DÉSACTIVER positional encoding pour vitesse maximale
    # (vous pouvez l'activer plus tard pour comparaison)
    # model.use_pos_encoding = False
    # print("\n⚠️  Positional encoding DÉSACTIVÉ pour vitesse maximale")
    print("    (réactivez avec model.use_pos_encoding = True)")
    
    print("="*60)
    print("✅ TransGNN Complet initialisé avec succès!")
    print(f"   - Échantillons: {sample_size} nœuds/nœud")
    print(f"   - Stratégie update: {update_strategy}")
    print(f"   - Blocs TransGNN: {args.block_num}")
    print("="*60)
    
    return model
import torch
import torch.nn as nn
import numpy as np

from model.MHTAN import multi_semantic

# for drug-target interaction link prediction task:
class DTI_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,    
                 num_edge_type,   
                 etypes_lists,    
                 in_dim,    
                 out_dim,    
                 num_heads,    
                 attn_vec_dim,    
                 aggregator='transformer',
                 attn_drop=0.2):   
        super(DTI_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.drug_layer = multi_semantic(num_metapaths_list[0],
                                                   etypes_lists[0],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   aggregator,
                                                   attn_drop,
                                                   use_minibatch=True)
        self.protein_layer = multi_semantic(num_metapaths_list[1],
                                                   etypes_lists[1],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   aggregator,
                                                   attn_drop,
                                                   use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads as multiple head outputs are concatenated together
        self.fc_drug = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_protein = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_drug.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_protein.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # obtain drug embedding and protein embedding
        h_drug = self.drug_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_protein = self.protein_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))

        logits_drug = self.fc_drug(h_drug)
        logits_protein = self.fc_protein(h_protein)
        return [logits_drug, logits_protein], [h_drug, h_protein]


class DTI_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 aggregator='transformer',
                 dropout_rate=0.2):
        super(DTI_lp, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])

        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x

        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # DTI_lp layers
        self.layer1 = DTI_lp_layer(num_metapaths_list,
                                     num_edge_type,
                                     etypes_lists,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     aggregator,
                                     attn_drop=dropout_rate)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # feature transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        [logits_drug, logits_protein], [h_drug, h_protein] = self.layer1(
            (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))

        return [logits_drug, logits_protein], [h_drug, h_protein]

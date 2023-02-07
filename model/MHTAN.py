import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


class single_semantic(nn.Module):
    def __init__(self,
                 etypes,    
                 out_dim,    
                 num_heads,   
                 aggregator='transformer',    
                 attn_drop=0.2,   
                 alpha=0.01,    # nn.LeakyReLU(alpha)
                 use_minibatch=True,  
                 attn_switch=True):
        super(single_semantic, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.aggregator = aggregator
        self.etypes = etypes
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch

        # metapath instance aggregator
        if aggregator == 'transformer':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)

        # single_semantic attention
        if self.attn_switch:
            self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        else:
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        # weight initialization
        if self.attn_switch:
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices = inputs

        edata = F.embedding(edge_metapath_indices, features)

        # metapath instance-level transformer
        if self.aggregator == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=128, dim_feedforward=2048, nhead=4)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            transformer_encoder=transformer_encoder.to(edata.device)
            hidden = transformer_encoder(edata.permute(1, 0, 2))
            hidden = self.rnn(torch.mean(hidden, dim=0)).unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  

        if self.attn_switch:
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  
            a1 = self.attn1(center_node_feat)
            a2 = (eft * self.attn2).sum(dim=-1)
            a = (a1 + a2).unsqueeze(dim=-1)
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)
        a = self.leaky_relu(a)
        g = g.to(torch.device('cuda'))
        g.edata.update({'eft': eft, 'a': a})

        # compute softmax normalized attention values
        self.edge_softmax(g)

        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']

        if self.use_minibatch:
            return ret[target_idx]
        else:
            return ret


class multi_semantic(nn.Module):
    def __init__(self,
                 num_metapaths,    # num_metapaths_list[0] or num_metapaths_list[1]
                 etypes_list,    # etypes_lists[0] or etypes_lists[1]
                 out_dim,    
                 num_heads,    
                 attn_vec_dim,    
                 aggregator='transformer',
                 attn_drop=0.2,    
                 use_minibatch=True):    
        super(multi_semantic, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch
        
        
        # aggregate semantic-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(single_semantic(etypes_list[i],
                                                                out_dim,
                                                                num_heads,
                                                                aggregator,
                                                                attn_drop=attn_drop,
                                                                use_minibatch=use_minibatch))

        # multi_semantic attention
        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, target_idx, metapath_layer in zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list, _ = inputs

            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, metapath_layer in zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h        

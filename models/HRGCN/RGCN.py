import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn


# def lstm_reducer(self, nodes):
#     """LSTM reducer
#     NOTE(zihao): lstm reducer with default schedule (degree bucketing)
#     is slow, we could accelerate this with degree padding in the future.
#     """
#     m = nodes.mailbox["m"].view(len(nodes), -1, self.out_dim*self.num_heads)  # (B, L, D)
#     # m = nodes.mailbox["m"]  # (B, L, D)
#     batch_size = m.shape[0]
#     h = (
#         m.new_zeros((1, batch_size, self.out_dim*self.num_heads)),
#         m.new_zeros((1, batch_size, self.out_dim*self.num_heads)),
#     )
#     _, (rst, _) = self.lstm(m, h)
#     return {"emb": rst.squeeze(0).view(len(nodes), self.out_dim, self.num_heads)}

class HeteroGraphConv(dglnn.HeteroGraphConv):

    def lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"].view(len(nodes), -1, self.out_dim*self.num_heads)  # (B, L, D)
        # m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self.out_dim*self.num_heads)),
            m.new_zeros((1, batch_size, self.out_dim*self.num_heads)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"emb": rst.squeeze(0).view(len(nodes), self.out_dim, self.num_heads)}
    
    def __init__(self, mods, in_dim, out_dim, aggregate=lstm_reducer):
        super().__init__(mods, aggregate)
        self.lstm = nn.LSTM(in_dim, out_dim, batch_first=True, bidirectional = True
)




class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class LSTMAggregator(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMAggregator, self).__init__()
        self.input_size = kwargs['input_size']
        self.hidden_size = kwargs['hidden_size']
        self.out_size = kwargs['out_size']
        self.max_len = kwargs['max_len']
        self.num_layer = kwargs['num_layer']
        self.drop = kwargs['drop']
        self.bn = kwargs['bn']
        self.cuda = kwargs['cuda']
        self.bi = kwargs['direction']
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layer, bidirectional=self.bi, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(self.max_len)
        self.dropout = nn.Dropout(self.drop)
        d_num = 1
        if self.bi:
            d_num = 2
        self.fc_in = nn.Linear(self.input_size, int(self.input_size/2))
        self.fc_out = nn.Linear(d_num * self.hidden_size * self.max_len, self.out_size)
        self.fc_out.requires_grad = True
        self.fc_in.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = True
        
    def forward(self, data, batch_masks=None, cluster=False): #tasks_embeds, task_sp_embeds=None
        if isinstance(data, tuple) and cluster==False:
            essen_data = data[0]
            add_data = data[1]
            essen_data = self.fc_in(essen_data) # ensure dimension consistancy bs * max_len * input_size/2
            bs = essen_data.shape[0]
            data = torch.cat((essen_data, add_data), dim=-1)
            #if not task_sp_embeds==None:
            #    task_res = torch.cat((tasks_embeds, task_sp_embeds), dim=-1)   
            #dummy_data = data[-1].repeat((1,1,2))
            #data_ = torch.cat((dummy_data, data), dim=1)
        else:
            bs = data.shape[0]  
        data = data.view(self.max_len, bs, -1) # out: the output embedding at the last LSTM layer, each corresponds to a input word           
        out, (h_n, h_c) = self.lstm(data) # hidden_cell: the h and c for all LSTM layers
        
        out = self.dropout(out)
        if self.bn:
            out = out.reshape((data.shape[1], self.max_len, -1))
            out = self.batch_norm(out)  
        res = torch.tanh(self.fc_out(out.reshape((bs, -1)))) # batch_size, hidden_size * bi_dir * max_len
        return res

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names, args, f_use_gnn):
        super().__init__()
        self.products_max_num = args.max_len
        self.bs = args.batch_size
        self.f_use_gnn = f_use_gnn
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if f_use_gnn:
            self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
            # self.classify = nn.Linear(hidden_dim*2, n_classes)
            self.lstm_agg = LSTMAggregator(input_size=hidden_dim, hidden_size=hidden_dim, max_len=self.products_max_num, num_layer=2, 
                out_size=args.embed_dim, bs=args.batch_size, drop=0.1, cuda=True, bn=False, direction=True)
        else:
            self.lstm_agg = LSTMAggregator(input_size=in_dim, hidden_size=hidden_dim, max_len=self.products_max_num, num_layer=2, 
                out_size=args.embed_dim, bs=args.batch_size, drop=0.1, cuda=True, bn=False, direction=True)
    def forward(self, g):
        h = g.ndata['feat']
        if self.f_use_gnn:
            h = self.rgcn(g, h)
        # test = (g.ndata['feat']['prod']).view(128*2, 45, 300)
        # unbatched_graph = dgl.unbatch(g)
        #batch_size, max_len, hidden_dim
            batched_feats = (h['prod']).view(-1, self.products_max_num, self.hidden_dim)
        else:
            batched_feats = (h['prod']).view(-1, self.products_max_num, self.in_dim)
        # batched_feats = (h['prod']).view(self.bs*2, self.products_max_num, 512)
        # batched_feats = (h['prod']).view(256*2, 25, 300)
        return self.lstm_agg(batched_feats)
        # with g.local_scope():
        #     g.ndata['h'] = h
        #     # Calculate graph representation by average readout.
        #     hg = 0
        #     for ntype in g.ntypes:
        #         hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
        #     hg, (h_n, h_c) = self.lstm(hg)
        #     hg = self.dropout(hg)
        #     return self.activation(self.classify(hg))
        


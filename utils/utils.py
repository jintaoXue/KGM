import numpy as np
import random
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import torch.nn as nn
import re
import os
from thop import profile # for evaluating model complexity
import json
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
from sklearn import datasets
import dgl
from dgl.data import DGLDataset 
from torch.autograd import Variable  
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from configs.config import Config, load_yaml
import yaml
class Dst(Dataset.Dataset):
    def __init__(self, data, word2id, tensor_max_len):
        self.data = [d[0][0] for d in data]
        self.masks = [d[0][1] for d in data]
        self.labels = [d[1] for d in data]
        self.sp_labels = [d[2] for d in data]
        self.abs_tlabels = [d[3] for d in data]
        self.word2id = word2id       
        self.tensor_max_len = tensor_max_len 
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.stack([ padding_single_tensor(d, self.tensor_max_len, self.word2id) for d in self.data[index] ]) # data is triples
        mask = torch.tensor(self.masks[index])
        label = torch.tensor(self.labels[index])
        sp_labels = self.sp_labels[index]
        abs_tlabels = self.abs_tlabels[index]
        return data, label, mask, sp_labels, abs_tlabels
    
class DGLDst(DGLDataset):
    def __init__(self, args, data, data_labels, id2term_embedding, tasks_embedding, task2id, name='KG', url=None, raw_dir=None, save_dir=None, hash_key=..., force_reload=False, verbose=False, transform=None):
        self.data = data
        self.labels = data_labels
        self.id2term_embedding = id2term_embedding 
        self.tasks_embedding = tasks_embedding
        self.task2id = task2id
        self.embed_dim = args.embed_dim
        self.args = args
        self.graphs = []
        self.num_etypes = None
        self.ntypes = None
        self.etypes = None
        self.per_data_len = args.max_len
        super().__init__(name, url, raw_dir, save_dir, hash_key, force_reload, verbose, transform)
      

    def process(self):
        # data is form of all_products_seg_ids : a tuple (sptial_terms id, relation id, product term id)
        graphs = []
        
        for i, all_products_seg_id in enumerate(self.data):
            data_len = len(all_products_seg_id)
            if  data_len > self.per_data_len:
                all_products_seg_id = all_products_seg_id[:self.per_data_len]
                data_len = self.per_data_len
            # data_len = self.per_data_len
            node_task = np.zeros(data_len)
            node_prod = np.arange(0, data_len)
            node_sp = np.arange(0, data_len)
            graph : dgl.heterograph.DGLGraph = dgl.heterograph({
                #sp: spatial terms (like sp-room space), spatial relation (like contains), product terms (like 2D panel)
                ('prod', 'in', 'task'): (node_prod, node_task),
                ('sp', 'rel', 'prod'): (node_sp, node_prod),
                ('sp', 'self', 'sp'): (node_sp, node_sp),
                ('prod', 'self', 'prod'): (node_prod, node_prod),
            })
            graph.nodes['task'].data['feat'] = torch.ones(1, self.embed_dim)
            all_products_seg_id = torch.tensor(all_products_seg_id)
            # if self.args.cuda:
            #     all_products_seg_id = all_products_seg_id.cuda()
            graph.nodes['prod'].data['feat'] = torch.index_select(self.id2term_embedding, 0, all_products_seg_id[:, 2])
            graph.nodes['sp'].data['feat'] = torch.index_select(self.id2term_embedding, 0, all_products_seg_id[:, 0])
            graph.edges['in'].data['feat'] = torch.ones(data_len, 1)
            graph.edges[('prod', 'self', 'prod')].data['feat'] = torch.ones(data_len, 1)
            graph.edges[('sp', 'self', 'sp')].data['feat'] = torch.ones(data_len, 1)
            graph.edges['rel'].data['feat'] = torch.index_select(self.id2term_embedding, 0, all_products_seg_id[:, 1])
            self.graphs.append(graph)
        self.num_etypes = len(graph.etypes)
        self.ntypes = graph.ntypes
        self.etypes = graph.etypes
        return 
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        label_id = self.task2id[self.labels[index]]
        label_embedding = self.tasks_embedding[label_id]
        return self.graphs[index], label_id, label_embedding

class DGLDst2(DGLDataset):
    def __init__(self, graphs=None, labels=None, labels_emb=None, masks=None, data_save_path=None, f_save_data = False, f_load_from_dir = False,
                name='KG', url=None, raw_dir=None, save_dir=None, hash_key=..., force_reload=False, verbose=False, transform=None):
        self.graphs = graphs
        self.labels = labels
        self.labels_emb = labels_emb
        self.masks = masks
        self.data_save_path = data_save_path
        super().__init__(name, url, raw_dir, save_dir, hash_key, force_reload, verbose, transform)
        if f_load_from_dir:
            self.data_load()
        if f_save_data:
            self.data_save()
        self.etypes = self.graphs[0].etypes
        self.ntypes = self.graphs[0].ntypes
    def process(self):
        # data is form of all_products_seg_ids : a tuple (sptial_terms id, relation id, product term id)
        return 

    def data_save(self):
        # save graphs and labels
        graph_path = os.path.join(self.data_save_path, self.name + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs)
        # save other information in python dict
        info_path = os.path.join(self.data_save_path, self.name + '_info.pkl')
        save_info(info_path, {'labels': self.labels, 'labels_emb': self.labels_emb, 'masks': self.masks, 'etypes': self.etypes, 'ntypes': self.ntypes})

    def data_load(self):
        # load processed data from directory `self.data_save_path`
        graph_path = os.path.join(self.data_save_path, self.name + '_dgl_graph.bin')
        self.graphs, _ = load_graphs(graph_path)
        info_path = os.path.join(self.data_save_path, self.name + '_info.pkl')
        info_dict = load_info(info_path)
        self.labels = info_dict['labels']
        self.labels_emb = info_dict['labels_emb']
        self.masks = info_dict['masks']
        self.etypes = info_dict['etypes']
        self.ntypes = info_dict['ntypes']

    def has_cache(self):
        # check whether there are processed data in `self.data_save_path`
        graph_path = os.path.join(self.data_save_path, self.name + '_dgl_graph.bin')
        info_path = os.path.join(self.data_save_path, self.name + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        return self.graphs[index], self.labels[index], self.labels_emb[index], self.masks[index]

def collate(samples):
    graphs, labels, labels_emb, masks = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    # batched_labels = torch.tensor(labels)
    return batched_graph, labels, labels_emb, masks

def reconstruct_terms_from_ids2(batch_products, all_term2id, word2id, id2word, rel2id):
    pad_id = word2id['PAD']
    rel_ids = [word2id[rn] for rn in list(rel2id.keys())]
    batch_sp_triples =[]
    for products in batch_products: # sp_batch_products: (batch_size,15,10); products: (15,10)
        current_products_triples =[]
        for product_data in products: # 1x10
            # find the rel location if any
            current_rel_id = None
            for rel_id in rel_ids: # note the rel_ids is the ids in the word-dict, not the relation dict
                try:
                    rel_loc = product_data.tolist().index(rel_id)
                    if not rel_loc==None:
                        current_rel_id = rel_id
                        break # one data (1x10) only has one rel
                except: # the whole row (1x10) is a term #print(product_data)
                    continue    
            if current_rel_id==None:
                term_lst = [ t for t in product_data.tolist() if not t==pad_id ]
                if len(term_lst)>0:
                    # term_1 = ( ' '.join([id2word[id] for id in term_lst]) ).strip() # join the words into terms and label them using all_term2idterm_lst
                    term_2_id = term_lst
                    # term_1_id = all_term2id[term_1]
                    term_1_id = [word2id['DUMMY_TASK']]
                    sp_rel = [word2id['constrains']]
                else:
                    continue  
            else:
                term_1_id = product_data.tolist()[ : rel_loc]
                # term_1 = ( ' '.join([id2word[id] for id in term_1_lst]) ).strip() # join the words into terms and label them using all_term2id
                # term_1_id = all_term2id[term_1]
                term_2_id = [ t for t in product_data.tolist()[rel_loc+1 : ] if not t==pad_id]
                # term_2 = ( ' '.join([id2word[id] for id in term_2_lst]) ).strip()
                # term_2_id = all_term2id[term_2]
                sp_rel = [current_rel_id]
            
            sp_triple = tuple((term_1_id, sp_rel, term_2_id))
            current_products_triples.append(sp_triple)
        batch_sp_triples.append(current_products_triples)  
    return batch_sp_triples


def get_sp_rel_raw_product_embs(batch_products, embed_matrix, word2id, id2word, args):
    pad_id = word2id['PAD']
    rel2id = args.rel_dic
    rel_ids = [word2id[rn] for rn in list(rel2id.keys())]
    batch_spatials_emb = []
    batch_relations_emb = []
    batch_raw_products_emb = []
    for products in batch_products: # sp_batch_products: (batch_size,15,10); products: (15,10)
        spatials_emb = []
        relations_emb = []
        raw_products_emb = []
        for product_data in products: # 1x10
            # find the rel location if any
            if len(product_data.shape) == 0:
                product_data = product_data.unsqueeze(0)
            current_rel_id = None
            for rel_id in rel_ids: # note the rel_ids is the ids in the word-dict, not the relation dict
                try:
                    rel_loc = product_data.tolist().index(rel_id)
                    if not rel_loc==None:
                        current_rel_id = rel_id
                        break # one data (1x10) only has one rel
                except: # the whole row (1x10) is a term #print(product_data)
                    continue    
            if current_rel_id==None:
                term_lst = [ t for t in product_data.tolist() if not t==pad_id ]
                if len(term_lst)>0:
                    # term_1 = ( ' '.join([id2word[id] for id in term_lst]) ).strip() # join the words into terms and label them using all_term2idterm_lst
                    term_2_id = term_lst
                    # term_1_id = all_term2id[term_1]
                    term_1_id = [word2id['DUMMY_TASK']]
                    sp_rel = [word2id['constrains']]
                else:
                    #mean this product_data have no info, we pad this as follows
                    term_2_id = [pad_id]
                    term_1_id = [pad_id]
                    sp_rel = [word2id['constrains']]
            else:
                term_1_id = product_data.tolist()[ : rel_loc]
                # term_1 = ( ' '.join([id2word[id] for id in term_1_lst]) ).strip() # join the words into terms and label them using all_term2id
                # term_1_id = all_term2id[term_1]
                term_2_id = [ t for t in product_data.tolist()[rel_loc+1 : ] if not t==pad_id]
                # term_2 = ( ' '.join([id2word[id] for id in term_2_lst]) ).strip()
                # term_2_id = all_term2id[term_2]
                sp_rel = [current_rel_id]
            spatial_emb = Variable(generate_one_data_embedding(term_1_id, embed_matrix, id2word, word2id, args))
            relation_emb =  Variable(generate_one_data_embedding(sp_rel, embed_matrix, id2word, word2id, args))
            raw_product_emb = Variable(generate_one_data_embedding(term_2_id, embed_matrix, id2word, word2id, args))
            # import copy
            # spatial_emb = generate_one_data_embedding(term_1_id, embed_matrix, id2word, word2id, args)
            # spatial_emb = copy.deepcopy(spatial_emb)
            # relation_emb =  generate_one_data_embedding(sp_rel, embed_matrix, id2word, word2id, args)
            # spatial_emb = copy.deepcopy(spatial_emb)
            # raw_product_emb = generate_one_data_embedding(term_2_id, embed_matrix, id2word, word2id, args)
            # spatial_emb = copy.deepcopy(spatial_emb)
            spatials_emb.append(spatial_emb)
            relations_emb.append(relation_emb)
            raw_products_emb.append(raw_product_emb)
        spatials_emb = torch.stack(spatials_emb).to('cpu')
        relations_emb = torch.stack(relations_emb).to('cpu')
        raw_products_emb = torch.stack(raw_products_emb).to('cpu')
        batch_spatials_emb.append(spatials_emb)
        batch_relations_emb.append(relations_emb)
        batch_raw_products_emb.append(raw_products_emb)
    return batch_spatials_emb, batch_relations_emb, batch_raw_products_emb

def get_graph_dataset(args, dataset, embed_matrix, id2word, word2id, sp_rel_prod_triples=None, id2task=None, save_path=None, name='KG', f_save_data = False, f_load_from_dir = False):
    graphs = []
    labels = []
    labels_emb = []
    masks = []
    for index, item in enumerate(dataset):            
        products, label, mask, sp_label, abs_tlabel = item
        products, label = products.unsqueeze(0), label.unsqueeze(0)
        # data = Variable(generate_batch_data(products, embed_matrix, id2word, word2id, args)).to('cpu') # convert batch data (indices) to numerical matrix         
        data = generate_batch_data(products, embed_matrix, id2word, word2id, args).to('cpu') # convert batch data (indices) to numerical matrix         
        label_ids = torch.stack([process_task_ids(args, bl, word2id, id2task) for bl in label]).unsqueeze(1)
        label_emb = Variable(generate_batch_data(label_ids, embed_matrix, id2word, word2id, args)).squeeze(1).to('cpu')
        data, label, label_emb = data.squeeze(0), label.squeeze(0), label_emb.squeeze(0)
        data_len = len(data)  

        node_task = np.zeros(data_len)
        node_prod = np.arange(0, data_len)
        if args.use_sp_data:
            node_sp = np.arange(0, data_len)
            graph : dgl.heterograph.DGLGraph = dgl.heterograph({
                #sp: spatial terms (like sp-room space), spatial relation (like contains), product terms (like 2D panel)
                ('prod', 'in', 'task'): (node_prod, node_task),
                ('sp', 'rel', 'prod'): (node_sp, node_prod),
                ('sp', 'self', 'sp'): (node_sp, node_sp),
                ('prod', 'self', 'prod'): (node_prod, node_prod),
            })
            graph.edges['rel'].data['feat'] = sp_rel_prod_triples[1][index]
            graph.nodes['sp'].data['feat'] = sp_rel_prod_triples[0][index]
            graph.edges[('sp', 'self', 'sp')].data['feat'] = torch.ones(data_len, 1)
        else:
            graph : dgl.heterograph.DGLGraph = dgl.heterograph({
                #sp: spatial terms (like sp-room space), spatial relation (like contains), product terms (like 2D panel)
                ('prod', 'in', 'task'): (node_prod, node_task),
                ('prod', 'self', 'prod'): (node_prod, node_prod),
            })
        graph.nodes['prod'].data['feat'] = data
        graph.edges[('prod', 'self', 'prod')].data['feat'] = torch.ones(data_len, 1)
        graph.nodes['task'].data['feat'] = torch.ones(1, 300)
        graph.edges['in'].data['feat'] = torch.ones(data_len, 1)
        # if (sp_rel_prod_triples[0][index]).shape != torch.Size([45, 300]):
        #     a= 1
        # if (sp_rel_prod_triples[1][index]).shape != torch.Size([45, 300]):
        #     a= 1
        graphs.append(graph)
        labels_emb.append(label_emb)
        labels.append(label)
        masks.append(mask)
    return DGLDst2(graphs=graphs, labels=labels, labels_emb=labels_emb, masks=masks, data_save_path=save_path, name=name, f_save_data = f_save_data, f_load_from_dir = f_load_from_dir)


'''perform complexity analysis, flops and paramter size'''
def complexity_analyze(model, data):
    '''compute FLOPs per unit'''
    input_size = list(data[0].size())[1:]
    input_size.insert(0, 1)
    inputs = torch.rand(input_size) # only consider 1 piece of data
    if data[1].cuda: # data[1]=args
        inputs = inputs.cuda()

    flops, params = profile(model, inputs, verbose=False)
    gflops_ = flops/1e+9
    params_ = params/1e+6
    return gflops_, params_

'''visualize features in the last hidden layer'''
def tsne_analyze(raw_data, hidden_states, num_class, col_path, tsne=None, threed=True):
    def get_color(col_path, num_class):
        pre_colors = dict()
        with open(col_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = [re.sub(r"[\', ]", '', l) for l in line.strip().split(':')]
                pre_colors.update({line[0] : line[1]})
        
        col_candiates = list(pre_colors.values())
        colors = []        
        for _ in range(num_class):
            randi = random.randint(0, num_class-1)
            tempc = col_candiates.pop(randi)
            colors.append(tempc)
        return colors
    
    try:
        hidden_states = np.array(hidden_states.cpu().numpy())
        raw_data = np.mean(np.array(raw_data.cpu().numpy()), axis=1)
    except:
        hidden_states = np.array(hidden_states.numpy())
        raw_data = np.mean(np.array(raw_data.numpy()), axis=1)
    
    if tsne==None:
        if threed:
            tsne = TSNE(n_components=3, init='pca')
        else:
            tsne = TSNE(n_components=2, init='pca')
    res = tsne.fit_transform(hidden_states)
    raw = tsne.fit_transform(raw_data)
    #res = (res - res.min()) / (res.max() - res.min()) # data normalization
    
    #colors = get_color(col_path, num_class)
    _, colors = datasets.make_s_curve(res.shape[0], random_state=0)
    fig = plt.figure(figsize=(16, 8))
    if threed:
        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(raw[:,0], raw[:,1], raw[:,2], c=colors, linewidths=0.5, marker='o', cmap=plt.cm.Spectral)
        ax.set_title('raw inputs')
        ax.view_init(4, -72)
        ax = fig.add_subplot(212, projection='3d')
        ax.scatter(res[:,0], res[:,1], res[:,2], c=colors, linewidths=0.5, marker='o', cmap=plt.cm.Spectral)
        ax.set_title('last hidden')
        ax.view_init(4, -72)
    else:
        ax = fig.add_subplot(2,1,1)
        ax.scatter(raw[:,0], raw[:,1], c=colors, linewidths=0.5, marker='o', edgecolors='k', cmap=plt.cm.Spectral)
        ax.set_title('raw inputs')
        ax = fig.add_subplot(2,1,2)
        ax.scatter(res[:,0], res[:,1], c=colors, linewidths=0.5, marker='o', edgecolors='k', cmap=plt.cm.Spectral)
        ax.set_title('last hidden')
    fig.tight_layout()
    #plt.show()
    

'''load pre-trained embedding matrix'''
def load_pretrained_embedding(corpus, embedding_file, split_char, embedding_dim, add_words=None, threshold=5):
    embeddings_dict = {}
    with open(embedding_file, 'r', encoding='UTF-8') as f: 
        for line in f:
            values = line.strip().split(split_char)
            if len(values) < threshold: # handling some special lines, e.g. tells us the number and dimension of the words in the file
                continue
            word = values[0] # values is a long list, the first element is the character/word, the others are embedding values
            embedding = np.asarray(values[1:], dtype='float32') # convert the embedding values
            embeddings_dict[word] = embedding # add above information to the dictionary
        if not add_words==None: # handle 'PAD' & 'DUMMY', randomly generate embeddings for them
            for sw in add_words:
                embeddings_dict.update({sw : np.random.uniform(low=-1.0, high=1.0, size=embedding_dim)}) 
        print('found {} word vectors in the entire pre-trained embeddings\n'.format(len(embeddings_dict)))
    
    word2id = {}
    corpus = add_words + corpus # special tokens are put at the beginning, PAD=0, DUMMY_TASK=1
    corpus_embedding_matrix = torch.Tensor(len(corpus), embedding_dim)
    for i, word in enumerate(corpus): # corpus: a list of all unique words in the training dataset
        word_embed = convert_data_to_tensor([word], embeddings_dict, dim=embedding_dim, pri=False)[0] # word is a single word
        corpus_embedding_matrix[i] = word_embed
        word2id.update({word:i})
    id2word = {v:k for k,v in word2id.items()}
    return corpus_embedding_matrix, word2id, id2word    

'''convert txt to embedding'''
def convert_data_to_tensor(txts, embedding_dic, mu=0, sigma=0.5, dim=50, pri=True):
    data_lst = []
    miss_count = 0
    for txt in txts:
        if len(txt.split(' '))>1: # in case txt is a phrase rather than a single word
            words = [w.strip() for w in txt.split(' ')]
            temp_tensors = [convert_data_to_tensor([w.strip()], embedding_dic, mu, sigma, dim)[0] for w in words]
            total_tesnor = temp_tensors[0]
            for i, t in enumerate(temp_tensors):
                if i==0:
                    continue
                total_tesnor = torch.cat((total_tesnor, t), 0)
            data_lst.append(torch.mean(total_tesnor, 0).numpy())
        else:
            if txt in list(embedding_dic.keys()):
                data_lst.append(embedding_dic[txt])
            else:
                data_lst.append(np.random.normal(mu, sigma, dim))
                miss_count += 1
    tensor = torch.FloatTensor(np.array(data_lst))
    if pri:
        print('found {} words without pre-trained embeddings'.format(miss_count))
    return tensor, miss_count

def padding_sequence(data, max_len, word2id, lst=False): # seq max_len is temporarily set as 15
    current_len = len(data)
    pad_res = data
    if lst:
        pad = [word2id['PAD']]
    else:
        pad = word2id['PAD']
    if current_len < max_len:
        while(len(pad_res) < max_len):
            pad_res.append(torch.tensor(pad))
    else:
        pad_res = pad_res[ : max_len]
    return pad_res

def padding_single_tensor(data, tensor_max_len, word2id): # tensor_max_len is temporarily set as 10
    try:
        current_len = len(data)
    except:
        current_len = 1
        data = torch.tensor([data])
    if len(data) > tensor_max_len:
        data = data[:tensor_max_len]
    pad = word2id['PAD']
    need_len = tensor_max_len - current_len
    pad_tensor = torch.tensor([pad] * need_len)
    pad_res = torch.cat((data, pad_tensor))
    return pad_res

def padding_tensors(batch_tensors, max_len, tensor_max_len, word2id):
    pad_res = []
    for tensors in batch_tensors:
        lag = max_len - tensors.shape[0]
        empty_tensor = torch.zeros((lag, tensor_max_len))
        empty_tensor.fill_(word2id['PAD']) 
        tensors = torch.cat((tensors, empty_tensor))
        pad_res.append(tensors)
    pad_res = torch.stack(pad_res)
    return pad_res

def split_list(dataset, frac_list, shuffle=False, random_state=None):
    frac_list = np.asarray(frac_list)
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)
    from itertools import accumulate
    return [
        [dataset[i] for i in indices[offset - length : offset]]
        for offset, length in zip(accumulate(lengths), lengths)
    ]

def raw_products_segmentation_to_ids(all_products, all_term2id, word2id, id2word, rel2id):
    # pad = 'PAD'
    # rel_ids = [word2id[rn] for rn in list(rel2id.keys())]
    batch_sp_triples =[]
    
    for products in all_products: 
        current_products_triples =[]
        for product_data in products: # 1x10
            # find the rel location if any
            current_rel_key = None
            term_lst = product_data.split()
            for rel_key in rel2id.keys(): # note the rel_ids is the ids in the word-dict, not the relation dict
                try:
                    rel_loc = term_lst.index(rel_key)
                    if not rel_loc==None:
                        current_rel_key = rel_key
                        break # one data (1x10) only has one rel
                except: # the whole row (1x10) is a term #print(product_data)
                    continue    
            if current_rel_key==None:
                if len(product_data)>0:   
                    term_2_id = all_term2id[product_data] # join the words into terms and label them using all_term2id
                    term_1_id = all_term2id['DUMMY_TASK']
                    sp_rel = all_term2id['constrains']
                else:
                    continue  
            else:
                term_1_lst = term_lst[ : rel_loc]
                term_1 = ( ' '.join([_term_1 for _term_1 in term_1_lst]) ) # join the words into terms and label them using all_term2id
                term_1_id = all_term2id[term_1]
                term_2_lst = term_lst[rel_loc+1 : ]
                term_2 = ( ' '.join([_term_2 for _term_2 in term_2_lst]) )
                term_2_id = all_term2id[term_2]
                sp_rel = all_term2id[current_rel_key]
            
            sp_triple = tuple((term_1_id, sp_rel, term_2_id))
            current_products_triples.append(sp_triple)
        batch_sp_triples.append(current_products_triples)  
    return batch_sp_triples

def generate_all_term_embedding(all_term2id, embed_matrix, id2word, word2id, args, sp_dir=None): # generate batch (embeddings) for both products and task labels
    res_embeds = []
    for i, terms in enumerate(all_term2id.keys()): # batch_products: batch_size, max_len (e.g., 10 or 15), tensor_max_len (e.g., 5 or 10)
        term_lst = terms.split(' ')
        temp_ids = torch.tensor([word2id[word] for word in term_lst])
        if args.cuda:
            temp_ids = temp_ids.cuda()
            embed_matrix = embed_matrix.cuda()
        temp_embds = torch.index_select(embed_matrix, 0, temp_ids) # embedding for the current product/term
        temp_embds = torch.mean(temp_embds, 0) # get average of the terms
        res_embeds.append(temp_embds) 
    res_embeds = torch.stack(res_embeds) # res: batch_size, max_len, embed_dim
    return res_embeds

'''process corpus and product/task entities'''
def generate_entities(data_path, rels=None, cluster=False):
    task_labels = []
    all_products = []
    all_terms = []
    deps = []
    with open(data_path, mode='r', encoding='utf-8') as f:
        for line in f:
            temp_task = line.strip().split('--')[0].strip()
            temp_task = re.sub(r'[^ a-zA-Z0-9]', '', temp_task) # remove characters that are not numbers/alphbets/spaces
            temp_task = re.sub(r'\s+', ' ', temp_task) # remove multiple spaces
            task_labels.append(temp_task.lower())
            temp_products = [p.strip().lower() for p in line.strip().split('--')[1].split(',') if not p=='']
            if cluster==True:
                temp_deps = temp_products[-3:]
                temp_products=temp_products[:-3]
                deps.append(temp_deps)
            for i, _ in enumerate(temp_products):
                temp_products[i] = re.sub(r'[^ a-zA-Z0-9]', '', temp_products[i])
                temp_products[i] = re.sub(r'\s+', ' ', temp_products[i])
            all_products.append(temp_products)
        all_terms.extend(task_labels) # terms are phrases & corpus are individual words
        all_terms.extend(all_products)
        all_terms = list(set(list(flat(all_terms))))
    f.close()
    
    if cluster==True:
        return task_labels, all_products, deps
    
    corpus = []
    for term in all_terms:
        corpus.extend(term.strip().split(' '))
    corpus = list(set(corpus))
    if not rels==None:
        for rel in list(rels.keys()):
            if not rel in corpus:
                corpus.append(rel)
    
    task2id = {}
    for i, t in enumerate(sorted(set(task_labels), key=task_labels.index)): 
        task2id.update({t:i})
    id2task = {v:k for k,v in task2id.items()}
    return task_labels, all_products, corpus, task2id, id2task

def generate_gnn_terms(all_products, word2id, relation2id, split_ch=' ', end_char='PAD', dummy_t='DUMMY_TASK'):
    unique_rels = list(relation2id.keys())
    all_term2id = { dummy_t:0 } # for easy extracting embedding of DUMMY-TASK
    for products in all_products:
        for words in products:
            if len(list(set(unique_rels).intersection(set(words.split(split_ch))))) == 0:
                current_term = (' '.join([w.strip() for w in words.split()])).strip()
                if not current_term in list(all_term2id.keys()):
                    all_term2id.update({current_term : len(all_term2id)})
            else:
                relation = list(set(unique_rels).intersection(set(words.split(split_ch))))[0]
                e1, relation, e2, all_term2id = parse_rel(words, relation, all_term2id)
                
    for k in list(relation2id.keys()):
        all_term2id.update( {k:len(all_term2id)} )
    all_term2id.update({ end_char : len(all_term2id) })
    id2all_term = {v:k for k,v in all_term2id.items()}    
    return all_term2id, id2all_term

def parse_rel(words, rel, all_term2id): # in this function, rel is not updated in all_term2id
    temps = [w.strip() for w in words.split()]
    rel_loc = temps.index(rel)
    e1 = (' '.join(temps[0:rel_loc])).strip()
    rel = temps[rel_loc]
    e2 = (' '.join(temps[rel_loc+1:])).strip()
    if not e1 in list(all_term2id.keys()):
        all_term2id.update({e1:len(all_term2id)})
    if not e2 in list(all_term2id.keys()):
        all_term2id.update({e2:len(all_term2id)})   
    return e1, rel, e2, all_term2id

def split_dataset(ratio, input_data):
    all_idx = list(range(0, len(input_data))) # all indices: 0, 1, 2... len(all_data)
    res_size = int(np.floor(len(all_idx) * ratio))
    count = 0
    select_idx = []
    assert len(all_idx) == len(input_data)
    
    while (count < res_size):
        temp_idx = random.randint(0, len(all_idx)-1)
        if not temp_idx in select_idx:
            select_idx.append(temp_idx)
            count += 1
    
    res_data = [input_data[i] for i in select_idx]
    remain_idx = [i for i in all_idx if not i in select_idx]
    remain_data = [input_data[i] for i in remain_idx]
    return res_data, remain_data

def generate_data(all_products, task_labels, word2id, task2id, max_len, sp_dic=None, id2word=None, data_split=None, tid2tid=None, rel_name='contains'): # generate all data with ids
    data = []
    labels = []
    sp_labels = []
    abt_labels = []
    try:
        rel_id = word2id[rel_name]
    except:
        rel_id = -1
    for i, products in enumerate(all_products): # each products is a list of single products
        temp_ids = []
        temp_spatials = []
        task_label = task2id[task_labels[i]] # note the task2id is the detail_tid, and the task_label is the detailed tid
        for product in products: # each product contains several words (including spatial relations)
            product_words_ids = torch.tensor([word2id[p] for p in product.strip().split(' ')])
            temp_ids.append(product_words_ids)
        temp_ids.insert(0, torch.tensor(word2id['DUMMY_TASK']))
        temp_ids = padding_sequence(temp_ids, max_len, word2id) # including padding or truncating
        masks = [0 if word2id['PAD'] in tid.numpy() else 1 for tid in temp_ids] # if the product is a PAD
        
        if not sp_dic==None:
            # if the product implies spatial info          
            sp_ids = [list(tid.numpy())[0:list(tid.numpy()).index(rel_id)] if rel_id in tid.numpy() else -1 for tid in temp_ids]
            
            current_sp_ids = []
            for sp_id in sp_ids:
                if not sp_id==-1:   
                    sp_label = sp_dic[(' '.join([id2word[id] for id in sp_id])).strip()]
                    current_sp_ids.append(sp_label)
                    current_sp_ids = list(np.unique(current_sp_ids))
                else:
                    continue
            sp_labels.append('+'.join([str(i) for i in current_sp_ids])) 
        else:
            sp_labels.append(-1)
            
        if not tid2tid==None:
            abt_labels.append(tid2tid[task_label]) # tid2tid {detail_tid : abstract_tid}
        else:
            abt_labels.append(-2)
        
        data.append((temp_ids, masks))
        labels.append(task_label)
        
    all_data = list(zip(data, labels, sp_labels, abt_labels))
    if not data_split==None:
        train_r, dev_r = data_split
        train_data, dev_data = split_dataset(train_r, all_data) # dev_data, test_data = split_data(dev_r, dev_test_data)
        return train_data, dev_data
    else:
        return all_data

def generate_batch_data(batch_rows, embed_matrix, id2word, word2id, args, sp_dir=None): # generate batch (embeddings) for both products and task labels
    batch_res = []
    for i, row in enumerate(batch_rows): # batch_products: batch_size, max_len (e.g., 10 or 15), tensor_max_len (e.g., 5 or 10)
        temp_res_embeds = []
        for k, words in enumerate(row): # row=products for one task; words=product, which is a product name containing certain words          
            init_word = id2word[int(words[0].numpy())]
            if not init_word == 'PAD':
                temp_ids = torch.tensor([int(word_id.numpy()) for word_id in words if not id2word[int(word_id.numpy())]=='PAD'])
            else:
                temp_ids = torch.tensor([word2id['PAD']]) # here only one element is ok
            if args.cuda:
                temp_ids = temp_ids.cuda()
                embed_matrix = embed_matrix.cuda()
            temp_embds = torch.index_select(embed_matrix, 0, temp_ids) # embedding for the current product/term
            temp_embds = torch.mean(temp_embds, 0) # get average of the terms
            temp_res_embeds.append(temp_embds) # products_embeds: list, each element is max_len (e.g., 10) * (embed_dim, )
        batch_res.append(torch.stack(temp_res_embeds)) # batch_es: list, each element is a tensor obtained by stacking products_embeds
    res = torch.stack(batch_res) # res: batch_size, max_len, embed_dim
    return res


def generate_one_data_embedding(words, embed_matrix, id2word, word2id, args, sp_dir=None): # generate batch (embeddings) for both products and task labels

    init_word = id2word[words[0]]
    if not init_word == 'PAD':
        temp_ids = torch.tensor([word_id for word_id in words if not id2word[word_id]=='PAD'])
    else:
        temp_ids = torch.tensor([word2id['PAD']]) # here only one element is ok
    if args.cuda:
        temp_ids = temp_ids.cuda()
        embed_matrix = embed_matrix.cuda()
    temp_embds = torch.index_select(embed_matrix, 0, temp_ids) # embedding for the current product/term
    embd = torch.mean(temp_embds, 0) # get average of the terms
    return embd

def generate_neg_data(products, labels, batch_masks, sp_labels, args):
    '''
        a key different between this and transE negative sampling is that this contaminate the task laebls, rather than the data
    '''
    neg_pos_ratio = args.neg_ratio
    if args.loss_func == 'soft_margin':
        neg_pos_ratio = 1
    last_idx = len(products)
    unique_labels = list(range(0, args.class_num))
    
    products = products.repeat((neg_pos_ratio, 1, 1))
    batch_masks = batch_masks.repeat((neg_pos_ratio, 1, 1))
    batch_masks = batch_masks.reshape((last_idx * neg_pos_ratio, -1, products.shape[1])).unsqueeze(1)
    sp_labels = sp_labels * neg_pos_ratio #repeat((neg_pos_ratio, 1, 1))
    #sp_labels = sp_labels.reshape((last_idx * neg_pos_ratio, -1, products.shape[1])).unsqueeze(1)    
    
    labels = labels.repeat((neg_pos_ratio))
    for i, l in enumerate(labels):
        if i<last_idx: # ensure the previous 'last_indx' data are true data
            continue
        else: # else, replace the taks label to generate negative data
            while(True):
                temp_label = random.choice(unique_labels)
                if not temp_label == l: # select a task label other than the original one
                    labels[i] = temp_label
                    break
    targets = torch.tensor([int(1)] * last_idx + [int(-1)] * (neg_pos_ratio-1) * last_idx) # -1 refers to the original data
    return products, labels, batch_masks, sp_labels, targets.unsqueeze(-1)

def generate_task_sp_embeds(sp_dir, id2task, all_term2id, gnn_out_entity, cuda): # generate spatial embeddings (based on spatial terms) for tasks
    task_sp_embeds = []
    for tid, sp_terms in sp_dir.items(): # a term is a phrase indicating the spatial space, a task can involve 2 or more spaces
        t_name = id2task[tid]
        temp_sp_ids = []
        for term in sp_terms:
            temp_sp_ids.append(all_term2id[term])
        
        temp_sp_ids = torch.tensor(temp_sp_ids)
        if cuda==True:
            temp_sp_ids = temp_sp_ids.cuda()
        temp_sp_embeds = torch.mean(torch.index_select(gnn_out_entity, 0, temp_sp_ids), dim=0)
        task_sp_embeds.append(temp_sp_embeds)
    task_sp_embeds = torch.stack(task_sp_embeds)
    return task_sp_embeds

def process_sp_info(sp_pth): # build spatial information directory i.e., {task_id:sp_name}
    task_sp_dir = {}
    with open(sp_pth, encoding='utf-8') as f:
        for line in f:
            t_name = line.strip().split('--')[0]
            t_name = re.sub(r'[^ a-zA-Z0-9]', '', t_name) # remove characters that are not numbers/alphbets/spaces
            t_name = re.sub(r'\s+', ' ', t_name).lower().strip()
            sp_info = [sp.lower() for sp in line.strip().split('--')[1].split(';')]
            task_sp_dir.update({t_name : sp_info})
    return task_sp_dir

def process_task_hier(t2t_pth, abs_tid, detail_tid, none_tag='NA'): # note the abs_tid and detail_tid can be the same (same granularity)
    task2task = {}
    tid2tid = {}
    if len(abs_tid)==len(detail_tid):
        tid_detial = {v:k for k,v in detail_tid.items()}
        task2task = {k:tid_detial[v] for k,v in abs_tid.items()}
        tid2tid = {v:detail_tid[k] for k,v in abs_tid.items()}
    else: 
        with open(t2t_pth) as f:
            for line in f:
                t_abs = line.strip().split('--')[0].strip() # in the txt file, the left is the abstract task, the right is the fine-grained task
                t_det = [t.strip() for t in line.strip().split('--')[1].split(';')]
                if not none_tag in t_det:
                    for t in t_det:
                        task2task.update({t:t_abs})
                        tid2tid.update({detail_tid[t] : abs_tid[t_abs]})
                else:
                    task2task.update({t_abs : t_abs}) # if there is no detailed task, then use t_abs to replace it 'the first t_abs'
                    tid2tid.update({detail_tid[t_abs] : abs_tid[t_abs]}) # detailed task id : abstract task id
    return task2task, tid2tid

def process_task_si(task_labels, task2sp, sp_dic):
    task_names = list(np.unique(task_labels))
    task2si = {}
    for tn in task_names:
        temp_sp_names = task2sp[tn]
        temp_sid = [ sp_dic[n] for n in temp_sp_names ]
        task2si.update({tn : temp_sid})
    return task2si

def process_task_ids(args, task_id, word2id, id2task): # convert textual task names to ids using word2id
    if type(task_id) == torch.Tensor:
        idx = task_id.item()
    else:
        idx = task_id   
    task_name = id2task[idx].strip()
    task_name_ids = torch.tensor([word2id[w] for w in task_name.split(' ')])
    task_name_ids = padding_single_tensor(task_name_ids, args.tensor_max_len, word2id) 
    return task_name_ids

def process_str_dics(dic_):
    res_dic = {}
    res_dic_reverse = {}
    for k, v in dic_.items():
        res_dic.update({k : int(v)})
        res_dic_reverse.update({int(v) : k})
    return res_dic, res_dic_reverse

def load_pretrain_mat(pth, dp): #'float32'
    embedding_file = os.path.join(os.getcwd(), pth)
    res_embed_mat = []
    with open(embedding_file, 'r', encoding='UTF-8') as f:
        for line in f:
            values = line.strip().split(' ')
            res_embed_mat.append(torch.tensor(np.asarray(values, dtype=dp)))
    res_embed_mat = torch.stack(res_embed_mat)
    return res_embed_mat

def load_x2id_dic(x2id_pth): # this dic record xx2id
    with open(os.path.join(os.getcwd(), x2id_pth), mode='r') as jf: 
        dics = json.load(jf)
        str_word2id = dics['word2id']
        word2id, id2word = process_str_dics(str_word2id)
        str_task2id = dics['task2id']
        task2id, id2task = process_str_dics(str_task2id)
        try:
            str_term2id = dics['term2id']
            all_term2id, _ = process_str_dics(str_term2id)
        except:
            all_term2id = None
        corpus = dics['corpus']
    jf.close()
    return corpus, word2id, id2word, task2id, id2task, all_term2id

def load_args(pth):
    args = argparse.ArgumentParser()
    args.add_argument("-ld", "--mlp_layer_dims", default=[256, 256], help="the dims of different layers for the MLP model")
    args.add_argument("-cf", "--cnn_filters", default=[(1, 8, 5), (1, 4, 25), (1, 2, 45)], help="CNN kenerls")
    args.add_argument("-rid", "--rel_dic", default={'contains':0, 'constrains':1}, help='relation2id')
    
    with open(os.path.join(os.getcwd(), pth), 'r') as af:
        for line in af:
            k = line.strip().split(':')[0].strip()
            v = line.strip().split(':')[1].strip()
            tp = line.strip().split(':')[-1].strip()
            if 'bool' in tp:
                v = True if v.lower() == 'true' else False
            elif 'int' in tp:
                v = int(v)
            elif 'float' in tp:
                v = float(v)
            elif 'str' in tp:
                pass
            else:
                continue
            args.add_argument('--'+k, default=v)
            #args_dic.update({k : v})
    args = args.parse_args() 
    return args

def one_hot(labels, batch_size, class_num):
    labels = labels.view(batch_size, 1)
    m_zeros = torch.zeros(batch_size, class_num)
    one_hot = m_zeros.scatter_(1, labels, 1)
    one_hot = one_hot.long() #print(one_hot.type(), ' ',one_hot[1:10])
    return one_hot

def softmax(x):
    row_max = np.max(x)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    res = x_exp / x_sum
    return res

def flatten(nest_list:list):
    return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]

def flat(input_lst):
    lst= []
    for i in input_lst:
        if type(i) is list:
            for j in i:
                lst.append(j)
        else:
            lst.append(i)
    return(lst)

def normalization(data, mode=1):
    if mode==1:
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma
    else:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
def write_tensor(tensor_, path):
    with open(os.path.join(os.getcwd(), path), mode='w') as f:
        np.savetxt(f, tensor_.cpu().detach().numpy())
    f.close()

def write_dev_data(dev_data, id2task, id2word, out_pth):
    with open(out_pth, mode='w') as f:
        for i, dev in enumerate(dev_data):
            task = id2task[dev[1]]
            f.write(task + '--')      
            tensor_products = dev[0][0]
            str_products = []
            for p in tensor_products:
                try:
                    p_name = ' '.join([id2word[s] for s in p.tolist()]).strip()
                except:
                    p_name = id2word[p.numpy().tolist()]
                    if p_name == 'PAD':
                        continue
                #print(p_name)
                f.write(p_name + ', ')
                str_products.append(p_name)
            f.write('\n')
    f.close() 
    
def generate_neg_data2(batched_graph, labels, labels_emb, tasks_embeds, batch_masks, args):
    '''
        a key different between this and transE negative sampling is that this contaminate the task laebls, rather than the data
    '''
    neg_pos_ratio = args.neg_ratio
    if args.loss_func == 'soft_margin':
        neg_pos_ratio = 1
    last_idx = len(labels)
    unique_labels = list(range(0, args.class_num))
    unbatched_graph = dgl.unbatch(batched_graph)
    unbatched_graph = unbatched_graph*neg_pos_ratio
    bg = dgl.batch(unbatched_graph)
    labels_emb = labels_emb*neg_pos_ratio
    labels = labels*neg_pos_ratio
    batch_masks = batch_masks*neg_pos_ratio
    for i, l in enumerate(labels):
        if i<last_idx: # ensure the previous 'last_indx' data are true data
            continue
        else: # else, replace the taks label to generate negative data
            while(True):
                temp_label = random.choice(unique_labels)
                if not temp_label == l: # select a task label other than the original one
                    labels_emb[i] = tasks_embeds[temp_label].to('cpu')
                    labels[i] = torch.tensor(temp_label)
                    break
    targets = torch.tensor([int(1)] * last_idx + [int(-1)] * (neg_pos_ratio-1) * last_idx) # -1 refers to the original data
    return bg, labels, labels_emb, batch_masks, targets.unsqueeze(-1)


def load_json_cfgs(save_dir: str, config_name: str = 'config.json') -> None:
        """Load the config from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.

        Raises:
            FileNotFoundError: If the config file is not found.
        """
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'The config file is not found in the save directory{save_dir}.',
            ) from error
        return Config.dict2config(kwargs)
    
def get_default_kwargs_yaml(path, algo: str, env_id: str = None) -> Config:
        """Get the default kwargs from ``yaml`` file.

        .. note::
            This function search the ``yaml`` file by the algorithm name and environment name. Make
            sure your new implemented algorithm or environment has the same name as the yaml file.

        Args:
            algo (str): The algorithm name.
            env_id (str): The environment name.
            algo_type (str): The algorithm type.

        Returns:
            The default kwargs.
        """
        # path = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(path, f'{algo}.yaml')
        print(f'Loading {algo}.yaml from {cfg_path}')
        kwargs = load_yaml(cfg_path)
        default_kwargs = kwargs['defaults']
        env_spec_kwargs = kwargs[env_id] if env_id in kwargs else None
        default_kwargs = Config.dict2config(default_kwargs)

        if env_spec_kwargs is not None:
            default_kwargs.recurisve_update(env_spec_kwargs)

        return default_kwargs

def save_json_config(save_path, config: Config) -> None:
        """Save the configuration to the log directory.

        Args:
            config (Config): The configuration to be saved.
        """
        # if self._maste_proc:
        #     self.log('Save with config in config.json', 'yellow', bold=True)
        with open(os.path.join(save_path, 'config.json'), encoding='utf-8', mode='w') as f:
            f.write(config.tojson())

def save_yaml_config(save_path, config: Config):
    # employee_dict={'employee': {'name': 'John Doe',  'age': 35,
    # 'job': {'title': 'Software Engineer','department': 'IT','years_of_experience': 10},
    # 'address': {'street': '123 Main St.', 'city': 'San Francisco','state': 'CA', 'zip': 94102}}}
    # print("The python dictionary is:")
    # print(employee_dict)
    file=open(os.path.join(save_path, 'config.yaml'),"w")
    yaml.dump(config.todict, file)
    file.close()
    # print("YAML file saved.")
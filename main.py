# -*- coding: UTF-8 -*-
import torch
import os
import argparse
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from utils.utils import *
import aggregators
import train_eval
import json
import pandas as pd
from drawing import *
from models.HRGCN.RGCN import HeteroClassifier, HeteroRGCN
from models.simpleHGN.simple_HGN import SimpleHGN
from utils.logger import Logger, init_log
from configs.config import Config, load_yaml
import setproctitle
import datetime
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-sp", "--use_sp_data", default=False, help="if use spatial info and GNN") # False
    args.add_argument("-fig", "--fineg", default=True, help="if use fine granularity")
    args.add_argument("-rid", "--rel_dic", default={'contains':0, 'constrains':1}, help='relation2id')
    args.add_argument("-md", "--full_mode", default="complex", help="use simple or complex aggregator") # simple OR complex
    args.add_argument("-mn", "--model_name", default="mean", help="model for aggregation") # mean sum pool
    args.add_argument("-tk", "--topk", default=6, help="the top class for computing hit@")
    args.add_argument("-cn", "--class_num", default=28, help="the number of task types")
    args.add_argument("-lr", "--learning_r", default=0.0001) #0.0001
    args.add_argument("-eph", "--embed_path", default="../embeddings/glove_300d.txt", help="word embedding file") # glove_50d.txt
    args.add_argument("-ed", "--embed_dim", default=300, help="the dimension of embeddings")
    args.add_argument("-en", "--epoch_num", default=3, help="the number of epochs")
    args.add_argument("-ds", "--data_split", default=(0.8, 0.2), help="data split for train, dev")
    args.add_argument("-ml", "--max_len", default=45, help="the maximum number of a sequence")
    args.add_argument("-tml", "--tensor_max_len", default=10, help="the maximum number of a tensor")
    args.add_argument("-te", "--train_emebd", default=False, help="if change the input embeddings")
    
    args.add_argument("-os", "--out_size", default=300, help="model output layer dimension")
    args.add_argument("-nr", "--neg_ratio", default=2, help="negative sample ratio") # 1 2 3
    args.add_argument("-bs", "--batch_size", default=256, help="training batch size")
    args.add_argument("-ls", "--loss_func", default='margin_loss', help="loss func") # margin_loss, soft_margin (if soft margin is applied, the neg_ratio=1)
    args.add_argument("-dp", "--dropout", default=0.1)
    args.add_argument("-cd", "--cuda", default=True) # False
    args.add_argument("-nw", "--num_worker", default=0, help="the number of workers for dataloader")
    args.add_argument("-bn", "--batch_norm", default=False)
    args.add_argument("-lgd", "--load_graph_data", default=False)
    args.add_argument("-sgd", "--save_graph_data", default=False)
    args.add_argument("-wandb", "--use_wandb", default=False)
    args.add_argument("-debug", "--debug", default=False)
    # MLP parameters
    args.add_argument("-ld", "--mlp_layer_dims", default=[512, 512], help="the dims of different layers for the MLP model")
    # LSTM parameters
    args.add_argument("-hs", "--lstm_hs", default=512, help="LSTM hidden layer dimension")
    args.add_argument("-nl", "--lstm_layer", default=2, help="LSTM hidden layer number")
    args.add_argument("-bin", "--direction", default=True, help="LSTM bi-direction used")
    # CNN parameters
    args.add_argument("-st", "--cnn_st", default=1, help="CNN stride")
    args.add_argument("-ci", "--cnn_inc", default=1, help="CNN input channel, by default, 1")
    args.add_argument("-cf", "--cnn_filters", default=[(1, 8, 5), (1, 4, 25), (1, 2, 45)], help="CNN kenerls")
    # BERT parameters
    args.add_argument("-bhd", "--bert_heads", default=4, help="bert head numbers")
    args.add_argument("-bml", "--bert_pos_dim", default=64, help="bert max position embedding dim")
    args.add_argument("-bid", "--bert_inter_dim", default=1024, help="bert intermediate dim for fully connection and recovery")
    args.add_argument("-bih", "--bert_hidden_dim", default=256, help="hidden layer size for bert encoder and pooler")
    args.add_argument("-bln", "--bert_layer_num", default=4, help="bert layers")
    args.add_argument("-btn", "--bert_name", default="albert", help="transformer model name") # albert  deberta
    args.add_argument("-ust", "--use_trans", default=False, help="whether the transformer toolkit is used") # False True
    # GNN parameters
    args.add_argument("-ug", "--use_gnn", default=False, help="if gnn is used")
    args.add_argument("-gch", "--gnn_conv_ch", default=3, help='fitlers for conv decoder')
    args.add_argument("-ghd", "--gnn_heads", default=2, help="gnn head layers")
    args.add_argument("-nn", "--nhop", default=1, help="n-hop neighbour")
    args.add_argument("-alp", "--gat_alpha", default=0.1, help='Leaky relu parameter')
    args = args.parse_args()
    return args
args = parse_args()
arg_dic = args.__dict__
with open(os.path.join(os.getcwd(), './pretrain_info/args_set.txt'), 'w') as af:
    for arg, value in arg_dic.items():
        af.writelines(arg + ' : ' + str(value) + ' : ' + str(type(value)) + '\n')

'''=====================================================ENTER THE MAIN CODE========================================================='''
if __name__ == '__main__':
    train_eval.setup_seed(20)        
    '''step 1: load embedding matrix and geneate train data'''    
    if args.use_sp_data:
        if args.fineg:
            train_pth = 'data/train_spatial_data_detail.txt'
            test_pth = 'data/test_spatial_data_detail.txt'
            sp_dict_pth = 'pretrain_info/sp_dir_detail.txt'
        else:
            train_pth = 'data/train_spatial_data.txt'
            test_pth = 'data/test_spatial_data.txt'
            sp_dict_pth = 'pretrain_info/sp_dir.txt'
        task2sp = process_sp_info(sp_dict_pth)
    else:
        if args.fineg:
            train_pth = 'data/train_data_detail.txt'
            test_pth = 'data/test_data_detail.txt'
        else:
            train_pth = 'data/train_data.txt'
            test_pth = 'data/test_data.txt'

    all_task_labels, all_products, corpus, task2id, id2task = generate_entities(os.path.join(os.getcwd(), train_pth), args.rel_dic)
    embed_matrix, word2id, id2word = load_pretrained_embedding(corpus, args.embed_path, ' ', args.embed_dim, add_words=['PAD','DUMMY_TASK'])
    sp_dic = {'none':0, 'plane':1, 'room space':2, 'door interface':3, 'window interface':4, 'wall plane':5, 'mep interface':6, 'ceiling plane':7,
              'floor plane':8}

    current_datetime = datetime.datetime.now()
    setproctitle.setproctitle("xjt_{}".format(args.model_name))
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # config = Config(**arg_dic)
    result_save_path = os.path.dirname(__file__) + "/graph_results"
    cfg_path = os.path.dirname(__file__) + "/configs" 
    # save_json_config(save_path=result_save_path, config=config)
    # save_yaml_config(save_path=result_save_path, config=config)
    cfg : Config = get_default_kwargs_yaml(cfg_path, args.model_name)
    cfg.log_dir = result_save_path
    cfg.use_sp_data = args.use_sp_data
    cfg.use_gnn = args.use_gnn
    if args.use_wandb:
        cfg.logger_cfgs.use_wandb = True
    if args.use_sp_data:
        data_name = "_with_spatial"
        task2si = process_task_si(all_task_labels, task2sp, sp_dic)
    else:
        data_name = "_no_spatial"
        task2si = None
    cfg.exp_name = args.model_name + data_name + '-' + formatted_datetime

    logger : Logger = init_log(cfg)
    if args.debug:
        train_data, dev_data = generate_data(all_products[:2000], all_task_labels[:2000], word2id, task2id, args.max_len, data_split=args.data_split, id2word=id2word, sp_dic=sp_dic)
    else:
        train_data, dev_data = generate_data(all_products, all_task_labels, word2id, task2id, args.max_len, data_split=args.data_split, id2word=id2word, sp_dic=sp_dic)

    # write_dev_data(dev_data, id2task, id2word, os.path.join(os.getcwd(), 'data/dev_spatial_data.txt'))
    train_dataset = Dst(train_data, word2id, args.tensor_max_len)
    # e.g., train_data: ([tensor(w_id_0, w_id_2), ... tensor(w_id_14, w_id_2)] , task_id)
    print('trainng data volume:', len(train_data))
    print('data path: ', train_pth)
    
    test_task_labels, test_products, test_terms, _, _ = generate_entities(os.path.join(os.getcwd(), test_pth))
    test_data_noisy = generate_data(test_products, test_task_labels, word2id, task2id, args.max_len, id2word=id2word, sp_dic=sp_dic) # for testing there is no data_split
    test_data_total = test_data_noisy + dev_data
    test_dataset_noisy = Dst(test_data_noisy, word2id, args.tensor_max_len)
    test_dataset_total = Dst(test_data_total, word2id, args.tensor_max_len)
    print('testing data noise added {}, and testing data total volume {}'.format(len(test_data_noisy), len(test_data_total)))

    # generate task label ids [63, 48, 77...] & process spatial information
    task_ids = torch.stack([process_task_ids(args, tid, word2id, id2task) for tid in list(id2task.keys())]).unsqueeze(1)
    tasks_embeds = generate_batch_data(task_ids, embed_matrix, id2word, word2id, args).squeeze(1) # generate task embedding matrix, row maps to task id

    #spatial and graph data
    graph_data_save_path = os.path.dirname(__file__) + '/graph_dataset'
    if args.load_graph_data:
        #warning data_save_path name cannot be _save_path or save_path
        graph_train_datasets : DGLDst2 = DGLDst2(data_save_path=graph_data_save_path, name='train' + data_name, f_save_data = False, f_load_from_dir = args.use_gnn)
    else:
        if args.use_sp_data:
            sp_rel_prod_triples = get_sp_rel_raw_product_embs(batch_products=train_dataset.data, embed_matrix=embed_matrix, word2id=word2id, id2word=id2word, args=args)
        else :
            sp_rel_prod_triples = None
        graph_train_datasets : DGLDst2 = get_graph_dataset(args, train_dataset, embed_matrix, id2word, word2id, sp_rel_prod_triples, 
                                                        id2task=id2task, save_path=graph_data_save_path, name = 'train' + data_name, f_save_data = args.save_graph_data, f_load_from_dir = False)
    # sp_rel_prod_triples = get_sp_rel_raw_product_embs(batch_products=train_dataset.data, embed_matrix=embed_matrix, word2id=word2id, id2word=id2word, args=args)
    # graph_train_datasets2 : DGLDst2 = get_graph_dataset(args, train_dataset, embed_matrix, id2word, word2id, sp_rel_prod_triples, 
    #                                                     id2task=id2task, save_path=save_path, name = 'train_sp_detail', f_save_data = args.save_graph_data, f_load_from_dir = False)
    '''step 2: create models'''
    # model, gnn_model = train_eval.select_model(args=args, corpus=corpus, all_products=all_products, word2id=word2id, id2word=id2word, 
    #                                            ent_embeds=embed_matrix, cluster_or_granu=False, all_term2id=None)
    if args.model_name == 'gcn_lstm':
        model = HeteroClassifier(in_dim=args.embed_dim, hidden_dim=512, n_classes=args.embed_dim, rel_names=graph_train_datasets.etypes, args=args, f_use_gnn = True)
    elif args.model_name == 'simple_hgn':
        model = SimpleHGN(edge_dim=args.embed_dim, num_etypes=len(graph_train_datasets.etypes), in_dim=args.embed_dim, hidden_dim=512, num_classes=args.embed_dim,
            num_layers=4, heads= [4, 4, 4, 1], feat_drop=0.1, negative_slope=0.1,
            residual=False, beta=0.05, ntypes=graph_train_datasets.ntypes)    
    elif args.model_name == 'gcn':
        model = HeteroRGCN(in_dim=args.embed_dim, hidden_dim=512, n_classes=args.embed_dim, rel_names=graph_train_datasets.etypes, args=args, f_use_gnn = args.use_sp_data)
    gnn_model = None

    '''step 3: train models''' 
    if args.full_mode == 'complex':
        train_loader = DataLoader.DataLoader(graph_train_datasets, batch_size=args.batch_size, collate_fn=collate, drop_last=False, shuffle=True)
        # train_loader = DataLoader.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_r)
        all_loss, res, _, train_df = train_eval.train_evaluate2(args, logger, model, optimizer, train_loader, embed_matrix, id2word, word2id, tasks_embeds, 
                            gnn=gnn_model, id2task=id2task, task2si=task2si, tid2tid=None, flag='train')
        
        dic_info_pth = 'pretrain_info/dics.json'
        dic_info={}
        if gnn_model is not None:
            model_pth = './pretrain_info/agg_gnn.pkl'
            gnn_pth = './pretrain_info/gnn.pkl'        
            torch.save({'model': model.state_dict()}, model_pth)
            torch.save({'model': gnn_model.state_dict()}, gnn_pth)
            write_tensor(res[0], 'pretrain_info/gnn_ent_embeds.txt')
            write_tensor(res[1], 'pretrain_info/gnn_rel_embeds.txt')
            dic_info.update({'term2id':gnn_model.all_term2id})
        else:
            model_pth = './pretrain_info/agg.pkl'
            torch.save({'model': model.state_dict()}, model_pth)
        
        dic_info.update({'word2id':word2id})
        dic_info.update({'id2word':id2word})
        dic_info.update({'task2id':task2id})
        dic_info.update({'id2task':id2task})
        dic_info.update({'corpus':corpus})
        with open(os.path.join(os.getcwd(), dic_info_pth), mode='w') as jf:
            json.dump(dic_info, jf)
        jf.close() 
        
        write_tensor(embed_matrix, 'pretrain_info/embed_mat.txt')
        write_tensor(tasks_embeds, 'pretrain_info/task_embed.txt')
        write_tensor(task_ids.squeeze(1), 'pretrain_info/task_ids.txt')
        
        all_loss = pd.DataFrame({'loss': [l.data.cpu().detach().numpy() for l in all_loss]})
        all_loss.to_csv('./results/loss.csv', mode='a')
        train_df.to_csv('./results/train_df.csv', mode='a')
        
    '''step 4: testing model'''
    optimizer = None
    #spatial and graph data
    if args.load_graph_data:
        graph_test_datasets : DGLDst2 = DGLDst2(data_save_path=graph_data_save_path, name= 'test' + data_name, f_save_data = False, f_load_from_dir = True)
    else:
        if args.use_sp_data:
            sp_rel_prod_triples = get_sp_rel_raw_product_embs(batch_products=train_dataset.data, embed_matrix=embed_matrix, word2id=word2id, id2word=id2word, args=args)
        else:
            sp_rel_prod_triples = None
        graph_test_datasets : DGLDst2 = get_graph_dataset(args, train_dataset, embed_matrix, id2word, word2id, sp_rel_prod_triples, 
                                                        id2task=id2task, save_path=graph_data_save_path, name = 'test' + data_name, f_save_data = args.save_graph_data, f_load_from_dir = False)
        
    test_loader_practical = DataLoader.DataLoader(graph_test_datasets, batch_size=args.batch_size, collate_fn=collate, drop_last=False, shuffle=True)
    _, res, test_df, _ = train_eval.train_evaluate2(args, logger, model, optimizer, test_loader_practical, embed_matrix, id2word, word2id, tasks_embeds, 
                        gnn=gnn_model, id2task=id2task, task2si=task2si, tid2tid=None, flag='evaluation')
    
    test_df.to_csv('./results/test_df.csv', mode='a')

    logger.close()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn
import os
import xlrd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interp
from itertools import cycle

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

def process_xls_rows(temp_rows):
    temp_hits = []
    for _, row in enumerate(temp_rows):
        #print(row)
        if any([t.ctype==1 for t in row]):
            continue
        else:
            temp_hits.append({'hit6': row[0].value, 'hit3':row[1].value, 'hit1':row[2].value})
    return temp_hits[:-1] # the last metric is the avg result


def process_xls_sheets(xls_pth, model):
    xls_file = xlrd.open_workbook(xls_pth)

    sp_df = data_transfer(process_xls_rows(xls_file.sheets()[0]), model+'-SP-G') 
    nosp_df = data_transfer(process_xls_rows(xls_file.sheets()[1]), model+'-NSP-G')
    
    sp_detail_df = data_transfer(process_xls_rows(xls_file.sheets()[2]), model+'-SP-D')
    nosp_detail_df = data_transfer(process_xls_rows(xls_file.sheets()[3]), model+'-NSP-D')
    
    model_df = pd.concat([sp_df, nosp_df])
    model_df_detail = pd.concat([sp_detail_df, nosp_detail_df])
    return model_df, model_df_detail


def add_pd(hits, hit_type, model):
    df = pd.DataFrame(columns=['h_values', 'model', 'hit'])
    for i, hit in enumerate(hits):
        df.loc[i] = [hit, model, hit_type]
        #df = df.append({'value':hit, 'model':model, 'hit':hit_type}, ignore_index=True)
    return df


def data_transfer(model_hits, model):
    df = pd.DataFrame(columns=['h_values', 'model', 'hit'])
    hit6 = []
    hit3 = []
    hit1 = []
    for _, data in enumerate(model_hits):
        hit6.append(data['hit6'])
        hit3.append(data['hit3'])
        hit1.append(data['hit1'])
    
    df6 = add_pd(hit6, 'hit@6', model)
    df3 = add_pd(hit3, 'hit@3', model)
    df1 = add_pd(hit1, 'hit@1', model)
    #df = pd.concat([df6, df3, df1])
    df = df.append(df6, ignore_index=True)
    df = df.append(df3, ignore_index=True)
    df = df.append(df1, ignore_index=True)
    return df


def box_plot_figure(df, figure_out_path, fig_size=(15, 10), detail=False):
    figsize = fig_size
    #fig, _ = plt.subplots()
    ax = seaborn.boxenplot(x='model', y='h_values', data=df, hue='hit', orient='v', linewidth=0.8, palette=seaborn.color_palette('deep', 10))
    
    plt.xlabel('models', fontsize=12)
    plt.ylabel('hit@N values', fontsize=12) 
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend(title='metrics', fontsize=12)
    
    if not detail:
        ax.axhline(y=0.743, c=flatui[0], ls='-', linewidth=1.2, zorder=0, clip_on=False) #hit6 sp
        ax.axhline(y=0.711, c=flatui[1], ls='-', linewidth=1.2, zorder=0, clip_on=False) #hit3 sp
        ax.axhline(y=0.572, c=flatui[2], ls='-', linewidth=1.2, zorder=0, clip_on=False) #hit1 sp
        ax.axhline(y=0.570, c=flatui[3], ls='--', linewidth=1.2, zorder=0, clip_on=False) #hit6 nsp
        ax.axhline(y=0.344, c=flatui[4], ls='--', linewidth=1.2, zorder=0, clip_on=False) #hit3 nsp
        ax.axhline(y=0.165, c=flatui[5], ls='--', linewidth=1.2, zorder=0, clip_on=False) #hit1 nsp
    else:
        ax.axhline(y=0.826, c=flatui[0], ls='-', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.707, c=flatui[1], ls='-', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.617, c=flatui[2], ls='-', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.601, c=flatui[3], ls='--', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.429, c=flatui[4], ls='--', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.208, c=flatui[5], ls='--', linewidth=1.2, zorder=0, clip_on=False)
        
    plt.savefig(figure_out_path, dpi=300)
    #plt.show()    


def loss_plot(xls_file, sheet_id, save_name=None, figsize=(7,5)):
    LSTM=False
    if sheet_id<2:
        LSTM=True
    
    if LSTM==True:
        loss_df = pd.DataFrame(columns=['Step', 'LSTM-SP', 'LSTM-NSP'])
    else:
        loss_df = pd.DataFrame(columns=['Step', 'DNN-SP', 'BERT-SP', 'DNN-NSP', 'BERT-NSP'])
    
    for i, row in enumerate(xls_file.sheets()[sheet_id]):
        if any([t.ctype==1 for t in row]):
            continue
        else:
            if LSTM==True:
                loss_df.loc[i] = [i, row[0].value, row[1].value]
            else:
                loss_df.loc[i] = [i, row[0].value, row[1].value, row[2].value, row[3].value]
    steps = list(loss_df['Step'].values)

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1, 1, 1)
    if LSTM==True:
        plt.plot(steps, loss_df['LSTM-SP'], flatui[0], label='LSTM-SP', linewidth=1.5)
        plt.plot(steps, loss_df['LSTM-NSP'], flatui[3], label='LSTM-NSP', linewidth=1.5)
    else:
        plt.plot(steps, loss_df['DNN-SP'], flatui[1], label='DNN-SP', linewidth=1.5)
        plt.plot(steps, loss_df['BERT-SP'], flatui[2], label='BERT-SP', linewidth=1.5)
        plt.plot(steps, loss_df['DNN-NSP'], flatui[4], label='DNN-NSP', linewidth=1.5)
        plt.plot(steps, loss_df['BERT-NSP'], flatui[5], label='BERT-NSP', linewidth=1.5)
        
    plt.legend(fontsize=12, loc='upper right')
    plt.xticks(np.linspace(0, 200, 9), rotation=0, size=16)
    if LSTM==True:
        plt.yticks(np.linspace(0.05, 0.1, 5), rotation=0, size=16)
    else:
        plt.yticks(np.linspace(0.1, 0.7, 7), rotation=0, size=16)
    plt.xlabel('training steps', fontsize=16)
    plt.ylabel('loss', fontsize=16)    
    
    if not save_name==None:
        plt.savefig(os.path.join(result_pth, save_name), dpi=300)
    #plt.show()


def generate_one_hot_mt(preds, n_classes):
    res_mt = np.zeros((preds.shape[0], n_classes))
    for i, p in enumerate(preds):
        res_mt[i][p] = 1
    return res_mt


def find_index(lst, fn): # find all indexes of elements in a list according to the conditional function fn
    res = []
    for i, x in enumerate(lst):
        if fn(x):
            res.append(i)
    return res


def find_figlabel(i):
    if i==0:
        config = 'CG-NSP'
    elif i==1:
        config = 'CG-SP'
    elif i==2:
        config = 'FG-NSP'
    else:
        config = 'FG-SP' 
    return config

def evaluate_curves(ytest, scores, n_classes, auc=True):
    def compute_macro(n_classes, x_dic, y_dic): # x is fpr, x is is recall
        all_x = np.unique(np.concatenate([x_dic[i] for i in range(n_classes)]))
        mean_y = np.zeros_like(all_x)
        for i in range(n_classes):
             mean_y += interp(all_x, x_dic[i], y_dic[i])
        
        mean_y /= n_classes # Finally average it and compute AUC
        x_dic['avg'] = all_x
        y_dic['avg'] = mean_y
        return x_dic['avg'], y_dic['avg']
    
    res_x = dict()
    res_y = dict()
    for i in range(n_classes):
        current_ytest = ytest[:, i].astype(int) # nd-array
        current_score = scores[:, i]        
        if auc:
            res_x[i], res_y[i], thresholds = roc_curve(current_ytest, current_score, drop_intermediate=True)
        else:
            res_x[i], res_y[i], thresholds = precision_recall_curve(current_ytest, current_score)
    
    if auc:
        res_x['avg'], res_y['avg'] = compute_macro(n_classes, res_x, res_y)  
    else:
        res_x['avg'], res_y['avg'] = compute_macro(n_classes, res_x, res_y)
    return res_x, res_y

def compute_roc_and_prs(res_df, n_classes, figure_out_path, roc_path, tf_path, pr_path, macro=True, save_data = False):
    ytest = res_df['ytest'].values
    ytest = generate_one_hot_mt(ytest, n_classes)
    scores = np.array(list(res_df['scores']))
    pr_scores = np.array(list(res_df['prscores']))
    
    '''compute macro or micro curves''' # the performance can be very high
    if macro==True:
        fpr, tpr = evaluate_curves(ytest, scores, n_classes, auc=True)
        precs, recalls = evaluate_curves(ytest, scores, n_classes, auc=False)
    else:
        fpr = dict()
        tpr = dict()
        precs = dict()
        recalls = dict()
        fpr['avg'], tpr['avg'], _ = roc_curve(ytest.ravel(), scores.ravel(), drop_intermediate=True)
        precs['avg'], recalls['avg'], _ = precision_recall_curve(ytest.ravel(), scores.ravel())
        precs['avg'], recalls['avg'], _ = precision_recall_curve(ytest.ravel(), pr_scores.ravel())
    
    roc_auc = dict()
    roc_auc['avg'] = auc(fpr['avg'], tpr['avg']) # add the macro-avg value at the end of the DF
    
    tfpr_df = pd.DataFrame({'fpr':fpr["avg"], 'tpr':tpr["avg"]})
    roc_df= pd.DataFrame(roc_auc.items(), columns=['key', 'value'])
    prre_df = pd.DataFrame({'precs':precs['avg'], 'recalls':recalls['avg']}) # this is the full complex version
    
    tfpr_df.to_csv(tf_path, index=None, mode='ab')
    roc_df.to_csv(roc_path, index=None, mode='ab')
    try:
        previous_predf = pd.read_csv(pr_path, header=None) # read the simplified version
    except:
        previous_predf = pd.DataFrame() # if the first csv is empty
    prre_df.to_csv(pr_path, index=None, mode='w') # write the full version into
    prre_df = pd.read_csv(pr_path, header=None, skiprows=lambda x: x > 0 and (x-1) % 25 != 0) # simplify and read
    try:
        new_predf = previous_predf.append(prre_df)
    except:
        new_predf = prre_df
    new_predf.to_csv(pr_path, index=None, header=False, mode='w')
    
    return fpr["avg"], tpr["avg"], roc_auc, precs["avg"], recalls["avg"]


def process_rocdfs(df, key, val):
    dfs = []
    start = 0 # no headers
    for i, row in df.iterrows():
        if not i==0 and (key in list(row) or val in list(row)):
            temp_df = df.iloc[start+1 : i]
            temp_df.columns = [key, val]
            dfs.append(temp_df)
            start = i    
        if i==len(df)-1:
            temp_df = df.iloc[start+1 : ]
            temp_df.columns = [key, val]
            dfs.append(temp_df)                            
    return dfs


def draw_roc(tf_path, roc_path, figure_out_path):    
    # get values
    roc_df = pd.read_csv(roc_path, header=None)
    roc_dfs = process_rocdfs(roc_df, 'key', 'value')
    tfpr_df = pd.read_csv(tf_path, header=None)
    tfpf_dfs = process_rocdfs(tfpr_df, 'fpr', 'tpr')
    
    plt.figure()
    for i, current_df in enumerate(tfpf_dfs): # length 4, element sequence: coarse-nosp, coarse-sp, fine-nosp, fine-sp
        fpr = np.array(current_df['fpr']).astype(float)
        tpr = np.array(current_df['tpr']).astype(float)
        temp_auc = np.around(float(roc_dfs[i].iloc[len(roc_dfs[i])-1]['value']), 3)
        config = find_figlabel(i)
        plt.plot(fpr, tpr, label='{} AUC={})'''.format(config, str(temp_auc)), linestyle=':', linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-class ROC')
    plt.legend(loc="lower right")        
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
    plt.savefig(figure_out_path, dpi=300)
    plt.show()


def draw_prcurve(pr_path, figure_out_path):
    pr_df = pd.read_csv(pr_path, header=None)
    pr_dfs = process_rocdfs(pr_df, 'precs', 'recalls')
    
    plt.figure()
    for i, current_df in enumerate(pr_dfs): # length 4, element sequence: coarse-nosp, coarse-sp, fine-nosp, fine-sp
        precs = np.array(current_df['precs']).astype(float)
        recalls = np.array(current_df['recalls']).astype(float)
        config = find_figlabel(i)
        plt.plot(recalls, precs, label=config, linestyle=':', linewidth=2)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('multi-class PR')
    plt.legend(loc='lower left')        

    plt.savefig(figure_out_path, dpi=300)
    plt.show()
    
def draw_train_metric_from_csv_res(loss_path, hit1_path, hit3_path, hit6_path):
    import matplotlib.pyplot as plt 
    import matplotlib.colors as mcolors 
    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
        # return plt.colormaps.get_cmap(name, n)
    cmap = get_cmap(22)
    color_inter_num = 6

    color_dict = {0: 'darkorange', 4: 'orange', 2: 'forestgreen', 3: 'dodgerblue', 1: 'palevioletred', 5:'blueviolet'}
    # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    styles_dict = plt.style.available
    

    # for i in range(0, self._num_action):
    #     x,y,yaw = _sdc_trj[i,:,0], _sdc_trj[i,:,1], _sdc_trj[i,:,2]
    #     plt.plot(x, y, '-', color=cmap(i), ms=5, linewidth=2)
    #     for j in range(0, horizon):
    #         plt.arrow(x[j], y[j], torch.cos(yaw[j])/5, torch.sin(yaw[j])/5, width=0.01, head_width=0.03, head_length=0.02, fc=cmap(i),ec=cmap(i))

    # plt.style.use('fast')
    plt.style.use('seaborn-v0_8-white')
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(10,10), dpi=100)
    # gs = gridspec(1,4, )
    # gs = fig.add_gridspec(1,4) 
    ax_1 = plt.subplot(241)
    ax_2 = plt.subplot(245)
    ax_3 = plt.subplot(242)
    ax_4 = plt.subplot(246)
    ax_5 = plt.subplot(243)
    ax_6 = plt.subplot(247)
    ax_7 = plt.subplot(244)
    ax_8 = plt.subplot(248)
    for ax in fig.get_axes():
        ax.grid(True)
    interval = 500
    '''loss curve plot'''
    df = pd.read_csv(loss_path, header=None)
    # all_items = df.iloc[0]
    #one column is one data
    one_data_len = len(df) - 1
    x_data = np.arange(0, one_data_len)
    ax_1.set_title('Loss\Without Spatial Information', fontsize = 12)
    # ax_1.set_xlabel('Number Agents', fontsize=15)
    ax_1.set_ylabel('Training loss', fontsize=15)
    ax_2.set_title('Loss\With Spatial Information', fontsize = 12)
    ax_2.set_xlabel('Steps', fontsize=15)
    ax_2.set_ylabel('Training loss', fontsize=15)
    for i in np.arange(0, len(df.columns)):
        one_data = df[i]
        data_name = one_data[0]
        data = np.array(one_data[1:]).astype(float)
        suffix = data_name.split('__')[-1]
        if data_name == "Step" or suffix == 'MIN' or suffix == 'MIN':
            continue 
        if "no_spatial" in data_name:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_1.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_1.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_1.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_1.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
        else:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_2.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_2.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_2.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_2.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
    '''hit1 curve plot'''
    df = pd.read_csv(hit1_path, header=None)
    # all_items = df.iloc[0]
    #one column is one data
    one_data_len = len(df) - 1
    x_data = np.arange(0, one_data_len)
    ax_3.set_title('Hit@1\Without Spatial Information', fontsize = 12)
    # ax_1.set_xlabel('Number Agents', fontsize=15)
    ax_3.set_ylabel('Hit@1', fontsize=15)
    ax_4.set_title('Hit@1\With Spatial Information', fontsize = 12)
    ax_4.set_xlabel('Steps', fontsize=15)
    ax_4.set_ylabel('Hit@1', fontsize=15)
    for i in np.arange(0, len(df.columns)):
        one_data = df[i]
        data_name = one_data[0]
        data = np.array(one_data[1:]).astype(float)
        suffix = data_name.split('__')[-1]
        if data_name == "Step" or suffix == 'MIN' or suffix == 'MIN':
            continue 
        if "no_spatial" in data_name:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_3.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_3.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_3.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_3.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
        else:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_4.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_4.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_4.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_4.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
    '''hit3 curve plot'''
    df = pd.read_csv(hit3_path, header=None)
    # all_items = df.iloc[0]
    #one column is one data
    one_data_len = len(df) - 1
    x_data = np.arange(0, one_data_len)
    ax_5.set_title('Hit@3\Without Spatial Information', fontsize = 12)
    # ax_1.set_xlabel('Number Agents', fontsize=15)
    ax_5.set_ylabel('Hit@3', fontsize=15)
    ax_6.set_title('Hit@3\With Spatial Information', fontsize = 12)
    ax_6.set_xlabel('Steps', fontsize=15)
    ax_6.set_ylabel('Hit@3', fontsize=15)
    for i in np.arange(0, len(df.columns)):
        one_data = df[i]
        data_name = one_data[0]
        data = np.array(one_data[1:]).astype(float)
        suffix = data_name.split('__')[-1]
        if data_name == "Step" or suffix == 'MIN' or suffix == 'MIN':
            continue 
        if "no_spatial" in data_name:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_5.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_5.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_5.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_5.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
        else:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_6.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_6.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_6.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_6.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
    '''hit6 curve plot'''
    df = pd.read_csv(hit6_path, header=None)
    # all_items = df.iloc[0]
    #one column is one data
    one_data_len = len(df) - 1
    x_data = np.arange(0, one_data_len)
    ax_7.set_title('Hit@6\Without Spatial Information', fontsize = 12)
    # ax_1.set_xlabel('Number Agents', fontsize=15)
    ax_7.set_ylabel('Hit@6', fontsize=15)
    ax_8.set_title('Hit@6\With Spatial Information', fontsize = 12)
    ax_8.set_xlabel('Steps', fontsize=15)
    ax_8.set_ylabel('Hit@6', fontsize=15)
    for i in np.arange(0, len(df.columns)):
        one_data = df[i]
        data_name = one_data[0]
        data = np.array(one_data[1:]).astype(float)
        suffix = data_name.split('__')[-1]
        if data_name == "Step" or suffix == 'MIN' or suffix == 'MIN':
            continue 
        if "no_spatial" in data_name:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_7.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_7.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_7.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_7.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
        else:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_8.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_8.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_8.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_8.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
    
    # c=[1,2,3,4]
    # labels = ['HGN-MC', 'RGCN', 'BiLSTM', 'RGCN-BiLSTM']
    # cmap = mcolors.ListedColormap(['darkorange','palevioletred','forestgreen','dodgerblue'])
    # norm = mcolors.BoundaryNorm([1,2,3,4,5],4)
    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # # cbar=plt.colorbar(sm, ticks=c, orientation='horizontal')
    # cbar = plt.colorbar(sm,ax=ax_8, orientation='horizontal', ticks=c)
    # cbar.set_ticklabels(labels)

    # pos = ax_4.get_position()
    # ax_4.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax_1.legend(
        loc='upper right', 
        bbox_to_anchor=(1.0, 0.9),
        ncol=1, 
        fontsize="10"
    )
    plt.show(block=False)
    fig.tight_layout()
    fig.savefig('{}.pdf'.format('./' + 'training_curves'), bbox_inches='tight')

    return



def draw_test_metric_from_csv_res(loss_path, hit1_path, hit3_path, hit6_path):


    pass


'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ###loss curves
    loss_results_path = os.path.dirname(__file__) + "/graph_results/aWandb" + "/wandb_train_loss.csv"
    hit1_results_path = os.path.dirname(__file__) + "/graph_results/aWandb" + "/wandb_hit1.csv"
    hit3_results_path = os.path.dirname(__file__) + "/graph_results/aWandb" + "/wandb_hit3.csv"
    hit6_results_path = os.path.dirname(__file__) + "/graph_results/aWandb" + "/wandb_hit6.csv"
    draw_train_metric_from_csv_res(loss_results_path, hit1_results_path, hit3_results_path, hit6_results_path)


    draw_test_metric_from_csv_res(loss_results_path, hit1_results_path, hit3_results_path, hit6_results_path)
    


    # draw_roc(os.path.join(os.getcwd(), 'results/tfr_df.csv'), 
    #          os.path.join(os.getcwd(), 'results/roc_df.csv'), 
    #          os.path.join(os.getcwd(), 'roc'))
    
    # draw_prcurve(os.path.join(os.getcwd(), 'results/pr_df.csv'),  
    #          os.path.join(os.getcwd(), 'prg'))
        
    # plot mapping performance
    # result_pth = os.path.join(os.path.join(os.getcwd(),'results'), 'record res')
    
    #bert_pth = os.path.join(result_pth, 'BERT_res.xls')
    #bert_df, bert_df_detail = process_xls_sheets(bert_pth, 'BERT') #bert_sp_df, bert_nosp_df, bert_sp_detail_df, bert_nosp_detail_df
    #
    #lstm_pth = os.path.join(result_pth, 'LSTM_res.xls')
    #lstm_df, lstm_df_detail = process_xls_sheets(lstm_pth, 'BiLSTM')
    #
    #dnn_pth = os.path.join(result_pth, 'DNN_res.xls')
    #dnn_df, dnn_df_detail = process_xls_sheets(dnn_pth, 'DNN')
    #
    # total_df = pd.concat([lstm_df, dnn_df, bert_df])
    #total_df_detail = pd.concat([lstm_df_detail, dnn_df_detail, bert_df_detail])
    #
    # total_pth = os.path.join(result_pth, 'total.png')
    #total_detail_pth = os.path.join(result_pth, 'total_detail.png')
    
    #box_plot_figure(total_df, total_pth)
    #box_plot_figure(total_df_detail, total_detail_pth, detail=True)
    
    # plot loss curves
    #loss_pth = os.path.join(result_pth, 'Loss.xls')
    #xls_file = xlrd.open_workbook(loss_pth)
    #loss_plot(xls_file, 0, save_name='lstm-g.png')
    #loss_plot(xls_file, 1, save_name='lstm-d.png')
    #loss_plot(xls_file, 2, save_name='other-g.png')
    #loss_plot(xls_file, 3, save_name='other-d.png')
    
    # tx0 = 0
    # tx1 = 25
    # ty0 = 0.055
    # ty1 = 0.10
    #
    # sx = [tx0, tx1, tx1, tx0, tx0]
    # sy = [ty0, ty0, ty1, ty1, ty0]
    # plt.plot(sx, sy,'purple', linewidth=2)
    #
    # axins = inset_axes(ax1, width=2, height=1.5, loc='center right')
    # axins.plot(steps, loss_df['LSTM-SP'], color=flatui[0], ls='-', linewidth=1.2)
    # axins.plot(steps, loss_df['LSTM-NSP'], color=flatui[3], ls='-', linewidth=1.2)
    # axins.axis([tx0, tx1, ty0, ty1])








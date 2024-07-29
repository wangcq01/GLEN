import os
import argparse
import numpy as np
import pandas as pd
import config
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader
from models_proposed_baseline import normalLSTMpertime,normalGRUpertime,two_Delegru,Retain_pertime,IMVTensorLSTM_pertime,shap_normalGRUpertime,lrp_normalGRUpertime
from newmodel_training import shap_lrp_new_credit_training2,credit_time
import torch

parser = argparse.ArgumentParser(description="credit default prediction")
parser.add_argument('--seed', type=int, default=555, help='The random seed')
parser.add_argument('--depth', type=int, default=8, help='The length of time series')
parser.add_argument('--input_dim_s', type=int, default=9, help='The dimension of shop features')
parser.add_argument('--input_dim_l', type=int, default=10, help='The dimension of loan features')
parser.add_argument('--lr', type=int, default=0.01, help='The learning rate')
parser.add_argument('--decay', type=int, default=0.001, help='The learning rate')
parser.add_argument('--output_dim', type=int, default=3, help='The dimension for output')
parser.add_argument('--num_classes', type=int, default=3, help='The dimension for output')
parser.add_argument('--n_units', type=int, default=16, help='The hidden size for Tensor LSTM')
parser.add_argument('--dataset', type=str, default='credit')
parser.add_argument('--save_dirs', type=str, default='zscore_pub_time_model', help='The dirs for saving results')
parser.add_argument('--batch_size', type=int, default=256, help='The batch size when training NN')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')
parser.add_argument('--save_models', type=bool, default=False, help='Whether save the training models')

args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:1")

    save_path = args.save_dirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #######load dataset, split input and y vairables
    data_path = r'D:\cq\transfer\Revise Experiment\Data\public'
    data = pd.read_csv(os.path.join(data_path, 'new_public_train.csv'))
    data=data.drop(['Credit_Score'],axis=1)
    #data = data.drop(['Credit_Mix'], axis=1)

    y = pd.read_csv(os.path.join(data_path, 'public_first_y.csv'))
    y.loc[y['Credit_Score'] == 'Good', 'Credit_Score'] = 1
    y.loc[y['Credit_Score'] == 'Poor', 'Credit_Score'] = 0
    y.loc[y['Credit_Score'] == 'Standard', 'Credit_Score'] = 2
    y['Credit_Score'] = y['Credit_Score'].astype('int64')
    data_idx = np.array(list(set(data['idx'])))
    N = len(set(data['idx']))

    cols = list(data.columns[1:])

    category_list = ['Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']
    for col in cols:
        if col not in category_list:
            data[col] = (data[col] - np.mean(data[col])) / (
              np.std(data[(col)]))


    #####分为五折，groupfold
    unique_idx = data['idx'].unique()
    group_kfold = GroupKFold(n_splits=5)
    # 参数范围
    short = 0
    for fold, (train_index, test_index) in enumerate(group_kfold.split(unique_idx, groups=unique_idx)):
        train_groups = unique_idx[train_index]
        test_groups = unique_idx[test_index]

        ##分为训练集，验证集，测试集
        x_train = data[data['idx'].isin(train_groups)]
        x_test =data[data['idx'].isin(test_groups)]
        y_train = y[y['idx'].isin(train_groups)]
        y_test = y[y['idx'].isin(test_groups)]

        # 再在训练集里面划分验证集
        train_id, val_id = train_test_split(train_groups, test_size=0.2, random_state=42)
        X_train = x_train[x_train['idx'].isin(train_id)]
        X_val = x_train[x_train['idx'].isin(val_id)]
        Y_train = y_train[y_train['idx'].isin(train_id)]
        Y_val = y_train[y_train['idx'].isin(val_id)]

        new_train = X_train[cols]
        new_train = np.array(new_train)
        new_train = torch.tensor(new_train)
        train_Y = Y_train

        new_val = X_val[cols]
        new_val = np.array(new_val)
        new_val = torch.tensor(new_val)
        val_Y = Y_val

        new_test = x_test[cols]
        new_test = np.array(new_test)
        new_test = torch.tensor(new_test)
        test_Y = y_test

        X_train_t = new_train.reshape(-1, args.depth, len(cols))
        X_val_t = new_val.reshape(-1, args.depth, len(cols))
        X_test_t = new_test.reshape(-1, args.depth, len(cols))

        y_train_t = torch.Tensor(train_Y['Credit_Score'].to_numpy()).reshape(-1, args.depth - 1)
        y_val_t = torch.Tensor(val_Y['Credit_Score'].to_numpy()).reshape(-1, args.depth - 1)
        y_test_t = torch.Tensor(test_Y['Credit_Score'].to_numpy()).reshape(-1, args.depth - 1)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size,
                                  shuffle=False, drop_last=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False,
                                drop_last=True)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size,
                                 shuffle=False, drop_last=True)

        model_list=['GLEN']
        for model_name in model_list:
            if model_name == 'GRU':
                model = normalGRUpertime(config.config(model_name, args), short).to(device)
            elif model_name == 'LSTM':
                model = normalLSTMpertime(config.config(model_name, args), short).to(device)
            elif model_name == 'GLEN':
                model = two_Delegru(config.config(model_name, args), short).to(device)
            elif model_name == 'Retain':
                model = Retain_pertime(config.config(model_name, args), short).to(device)
            elif model_name == 'IMV_tensor':
                model = IMVTensorLSTM_pertime(config.config(model_name, args), short).to(device)
            elif model_name == 'SHAP':
                model = shap_normalGRUpertime(config.config(model_name, args), short).to(device)
            elif model_name=='LRP':
                model = lrp_normalGRUpertime(config.config(model_name, args), short).to(device)
            else:
                ModuleNotFoundError(f'Module {model_name} not found')
            print(f'Training model {model_name}')
            print(f'Num of trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
            #shap_lrp_new_credit_training2(model, model_name, train_loader, val_loader, test_loader, args, device, fold,X_train_t,X_test_t)
            credit_time(model,model_name,train_loader,val_loader,test_loader,args,device,fold)

#####compute sufficiency, comprehensevies
import numpy as np
from sklearn.metrics import accuracy_score
from typing import List, Dict, Any
from models_proposed_baseline import evaluate_normalGRUpertime
import config
import os
import argparse
import pandas as pd
import torch

##得到所有feature的classification score,需要softmax 然后在得到删除feature之后的score， 在相减，先在样本求均值，在在五组实验球均值


def sufficiency_comprehensiveness(model, data,s1_data,s2_data,s3_data,c1_data,c2_data,c3_data, labels):

    s1_scores = []
    s2_scores=[]
    s3_scores=[]

    c1_scores = []
    c2_scores = []
    c3_scores = []

    for i in range(len(data)):
    #cfor i in range(3):
        original_prob = model(data[i].unsqueeze(0).float())  #shape 10,5 time_depth, multiclass
        ##the biggest prob and index
        original_index=torch.argmax(original_prob,dim=1)
        original_value=original_prob[torch.arange(original_prob.size(0)),original_index]
        s1_prob=model(s1_data[i].unsqueeze(0).float())
        s1_value=s1_prob[torch.arange(s1_prob.size(0)),original_index]
        s1_diff=original_value-s1_value  #shape tensor(10)
        s1_scores.append(s1_diff)

        s2_prob = model(s2_data[i].unsqueeze(0).float())
        s2_value = s2_prob[torch.arange(s2_prob.size(0)), original_index]
        s2_diff = original_value - s2_value  # shape tensor(10)
        s2_scores.append(s2_diff)

        s3_prob = model(s3_data[i].unsqueeze(0).float())
        s3_value = s3_prob[torch.arange(s3_prob.size(0)), original_index]
        s3_diff = original_value - s3_value  # shape tensor(10)
        s3_scores.append(s3_diff)

        c1_prob = model(c1_data[i].unsqueeze(0).float())
        c1_value = c1_prob[torch.arange(c1_prob.size(0)), original_index]
        c1_diff = original_value - c1_value  # shape tensor(10)
        c1_scores.append(c1_diff)

        c2_prob = model(c2_data[i].unsqueeze(0).float())
        c2_value = c2_prob[torch.arange(c2_prob.size(0)), original_index]
        c2_diff = original_value - c2_value  # shape tensor(10)
        c2_scores.append(c2_diff)

        c3_prob = model(c3_data[i].unsqueeze(0).float())
        c3_value = c3_prob[torch.arange(c3_prob.size(0)), original_index]
        c3_diff = original_value - c3_value  # shape tensor(10)
        c3_scores.append(c3_diff)

    s1_scores = torch.stack(s1_scores).squeeze(1)
    s1_score=s1_scores.mean(dim=1).mean(dim=0)

    s2_scores = torch.stack(s2_scores).squeeze(1)
    s2_score=s2_scores.mean(dim=1).mean(dim=0)

    s3_scores = torch.stack(s3_scores).squeeze(1)
    s3_score=s3_scores.mean(dim=1).mean(dim=0)

    c1_scores = torch.stack(c1_scores).squeeze(1)
    c1_score=c1_scores.mean(dim=1).mean(dim=0)

    c2_scores = torch.stack(c2_scores).squeeze(1)
    c2_score=c2_scores.mean(dim=1).mean(dim=0)

    c3_scores = torch.stack(c3_scores).squeeze(1)
    c3_score=c3_scores.mean(dim=1).mean(dim=0)


    return s1_score,s2_score,s3_score,c1_score,c2_score,c3_score
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
parser.add_argument('--save_dirs', type=str, default='pub_evaluation', help='The dirs for saving results')
parser.add_argument('--batch_size', type=int, default=256, help='The batch size when training NN')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')
parser.add_argument('--save_models', type=bool, default=False, help='Whether save the training models')

args = parser.parse_args()

if __name__ == "__main__":
    # rationales = [...] (your rationales here)
    # labels = np.array(...) (your labels here)
    short=0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:1")

    data_path = r'D:\cq\transfer\Revise Experiment\Data\public'
    data = pd.read_csv(os.path.join(data_path, 'new_public_train.csv'))
    data = data.drop(['Credit_Score'], axis=1)
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
            #data[col] = (data[col] - np.min(data[col])) / (
                    #np.max(data[(col)]) - np.min(data[col]))
            data[col] = (data[col] - np.mean(data[col])) / (
                np.std(data[(col)]))



    train_idx = np.random.choice(data_idx, int(0.75 * N), replace=False)
    data_idx = data_idx[~np.in1d(data_idx, train_idx)]
    val_idx = np.random.choice(data_idx, int(0.15 * N), replace=False)
    data_idx = data_idx[~np.in1d(data_idx, val_idx)]
    test_idx = data_idx
   #1926 个customer
    test_X = data.loc[data['idx'].isin(test_idx), :]
    test_X = test_X[cols]
    new_test = np.array(test_X)
    new_test = torch.tensor(new_test)
    test_Y = y.loc[y['idx'].isin(test_idx), :]

    X_test_t = new_test.reshape(len(test_idx), args.depth, len(cols))
    y_test_t = torch.Tensor(test_Y['Credit_Score'].to_numpy()).reshape(len(test_idx), args.depth - 1)


###取feature
    s1_mask=torch.zeros_like(X_test_t)
    s1_mask[:,:,0]=1
    s1_mask[:, :,18] = 1
    s1_data=X_test_t*s1_mask

    s2_mask = torch.zeros_like(X_test_t)
    s2_mask[:, :, 0] = 1
    s2_mask[:, :, 1] = 1
    s2_mask[:, :, 18] = 1
    s2_mask[:, :, 14] = 1
    s2_data = X_test_t * s2_mask

    s3_mask = torch.zeros_like(X_test_t)
    s3_mask[:, :, 0] = 1
    s3_mask[:, :,1] = 1
    s3_mask[:, :, 5] = 1
    s3_mask[:, :, 8] = 1
    s3_mask[:, :, 4] = 1
    s3_mask[:, :, 15] = 1
    s3_mask[:, :, 14] = 1
    s3_mask[:, :, 18] = 1
    s3_mask[:, :, 16] = 1
    s3_mask[:, :, 9] = 1
    s3_data = X_test_t * s3_mask


    c1_mask = torch.ones_like(X_test_t)
    c1_mask[:, :, 18] = 0
    c1_mask[:, :, 0] = 0
    c1_data = X_test_t * c1_mask

    c2_mask = torch.ones_like(X_test_t)
    c2_mask[:, :,1] = 0
    c2_mask[:, :, 0] = 0
    c2_mask[:, :,18] = 0
    c2_mask[:, :, 14] = 0
    c2_data = X_test_t * c2_mask

    c3_mask = torch.ones_like(X_test_t)
    c3_mask[:, :, 0] = 0
    c3_mask[:, :, 1] = 0
    c3_mask[:, :, 5] = 0
    c3_mask[:, :, 8] = 0
    c3_mask[:, :, 4] = 0
    c3_mask[:, :,14] = 0
    c3_mask[:, :, 9] = 0
    c3_mask[:, :, 15] = 0
    c3_mask[:, :, 16] = 0
    c3_mask[:, :, 18] = 0
    c3_data = X_test_t * c3_mask

    model = evaluate_normalGRUpertime(config.config('GRU', args), short)
    model.load_state_dict(torch.load('4_credit_predict.pt'))
    s1,s2,s3,c1,c2,c3= sufficiency_comprehensiveness(model, X_test_t,s1_data,s2_data,s3_data,
                                                                   c1_data,c2_data,c3_data,y_test_t)
    print(f"Sufficiency: ",s1,s2,s3 )
    print(f"Comprehensiveness:", c1,c2,c3)

    #sufficiency 应该是越小越好，compre是越越好



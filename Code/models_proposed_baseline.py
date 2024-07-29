from torch import nn, Tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from typing import Optional
import shap

def _estimate_alpha1(feature_reps, targets):
    '''
    alpha parameters OLS estimation given projected input features and targets.

    Params:
    - feature_reps: array-like of shape (bs, T, d, units)
    - targets: array-like of shape (bs, T, units)

    returns:
    - un-normalised alpha weights: array-like of shape (bs, T, d)
    '''

    X_T, X = feature_reps, feature_reps.permute(0,2,1)
    #c = torch.bmm(X_T, X).detach().cpu().numpy()
    # X 维度bs, units, d 最后两个维度进行转置
    try:
        X_TX_inv = torch.linalg.inv(torch.bmm(X_T, X))
    except Exception as ex:
        print('exception')
        print(ex)
        X_TX_inv = torch.linalg.pinv((torch.bmm(X_T, X)+1e-8))
        #X_TX_inv =torch.from_numpy( np.linalg.pinv(c)).to(device)
    X_Ty = torch.bmm(X_T, targets.unsqueeze(-1))
    # Compute likely scores
    alpha_hat = torch.bmm(X_TX_inv, X_Ty)
    return alpha_hat

#不分解的gru
class two_Delegru(torch.jit.ScriptModule):
    def __init__(self, Delegru_params, short):
        super(two_Delegru, self).__init__()
        self.input_dim_s=Delegru_params['input_dim_s']
        self.input_dim_l = Delegru_params['input_dim_l']
        self.n_units=Delegru_params['n_units']
        self.depth = Delegru_params['time_depth']
        self.output_dim = Delegru_params['output_dim']
        self.num_classes = Delegru_params['num_classes']
        self.N_units = Delegru_params['n_units']
        self.bias = Delegru_params['bias']
        self.short=short

        #shop tensor LSTM parameter
        #torch.set_default_dtype(torch.float64)
        self.U_r_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.U_z_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.U_h_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.W_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.W_z_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.W_h_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.B_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units))
        self.B_z_s = nn.Parameter(torch.Tensor(self.input_dim_s, self.n_units))
        self.B_h_s = nn.Parameter(torch.Tensor(self.input_dim_s, self.n_units))

        #shop vanilla LSTM parameter
        self.u_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units) )  # 正态分布
        self.w_r_s = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.b_r_s = nn.Parameter(torch.zeros(self.N_units))
        self.w_z_s = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_z_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units) )
        self.b_z_s = nn.Parameter(torch.zeros(self.N_units))
        self.w_h_s = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_h_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units) )
        self.b_h_s = nn.Parameter(torch.zeros(self.N_units))
        self.w_p_s = nn.Parameter(torch.randn(self.N_units, self.output_dim))
        self.b_p_s = nn.Parameter(torch.zeros(self.output_dim))

        #loan tensor LSTM parameter
        self.U_r_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.U_z_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.U_h_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.W_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.W_z_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.W_h_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.B_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units))
        self.B_z_l = nn.Parameter(torch.Tensor(self.input_dim_l, self.n_units))
        self.B_h_l = nn.Parameter(torch.Tensor(self.input_dim_l, self.n_units))

        #loan vanilla LSTM parameter

        self.u_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )  # 正态分布
        self.w_r_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.b_r_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_z_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_z_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )
        self.b_z_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_h_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_h_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )
        self.b_h_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_p_l = nn.Parameter(torch.randn(self.N_units, self.output_dim))
        self.b_p_l = nn.Parameter(torch.zeros(self.output_dim))

        self.classify = torch.nn.Linear(self.N_units*2, self.output_dim, bias=self.bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B_f' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'b_f' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'B' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                #nn.init.orthogonal_(param.data)
                nn.init.xavier_uniform_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                #nn.init.orthogonal_(param.data)
                nn.init.xavier_uniform_(param.data)

    def forward(self, x,target, device):
        H_tilda_t_s = (torch.zeros(x.shape[0], self.input_dim_s, self.n_units)).to(device)
        H_tilda_t_l = (torch.zeros(x.shape[0], self.input_dim_l, self.n_units)).to(device)

        # 每个时刻对应所有feature的hidden
        h_tilda_t_s = torch.zeros(x.shape[0],  self.N_units).to(device)
        h_tilda_t_l = torch.zeros(x.shape[0], self.N_units).to(device)

        unorm_list_s = torch.jit.annotate(list[Tensor], [])
        unorm_list_l = torch.jit.annotate(list[Tensor], [])
        weights_list = torch.jit.annotate(list[Tensor], [])

        pred_list = torch.jit.annotate(list[Tensor], [])
        prob_list = torch.jit.annotate(list[Tensor], [])
        weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
        loss_CE = torch.as_tensor(0.0)
        #print('new begin')
        for t in range(self.depth-1):
            #先算shop
            # eq 5
            R_tilda_t_s = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_s, self.W_r_s) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :self.input_dim_s].unsqueeze(1), self.U_r_s) + self.B_r_s)
            Z_tilda_t_s = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_s, self.W_z_s) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :self.input_dim_s].unsqueeze(1), self.U_z_s) + self.B_z_s)
            G_tilda_t_s = torch.tanh(torch.einsum("bij,ijk->bik",(H_tilda_t_s * R_tilda_t_s), self.W_h_s) + \
                                      torch.einsum("bij,jik->bjk", x[:, t,:self.input_dim_s ].unsqueeze(1), self.U_h_s) + self.B_h_s)

            H_tilda_t_s = Z_tilda_t_s * H_tilda_t_s+(1-Z_tilda_t_s)*G_tilda_t_s #shape batch, feature, dim

            #所有lstm的关系 shop
            r_t_s = torch.sigmoid(((x[:, t, :self.input_dim_s] @ self.u_r_s) + (h_tilda_t_s @ self.w_r_s) + self.b_r_s))
            z_t_s = torch.sigmoid(((x[:, t, :self.input_dim_s] @ self.u_z_s) + (h_tilda_t_s @ self.w_z_s) + self.b_z_s))
            g_t_s = torch.tanh(((x[:, t, :self.input_dim_s] @ self.u_h_s) + ((h_tilda_t_s *r_t_s) @ self.w_h_s) + self.b_h_s))
            h_tilda_t_s = z_t_s * h_tilda_t_s +(1-z_t_s)*g_t_s

            #tensor lstm loan
            R_tilda_t_l = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_l, self.W_r_l) + \
                                        torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                     self.U_r_l) + self.B_r_l)
            Z_tilda_t_l = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_l, self.W_z_l) + \
                                        torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                     self.U_z_l) + self.B_z_l)
            G_tilda_t_l = torch.tanh(torch.einsum("bij,ijk->bik", (H_tilda_t_l * R_tilda_t_l), self.W_h_l) + \
                                     torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                  self.U_h_l) + self.B_h_l)

            H_tilda_t_l = Z_tilda_t_l * H_tilda_t_l + (1 - Z_tilda_t_l) * G_tilda_t_l  # shape batch, feature, dim

            # 所有lstm的关系 shop
            r_t_l = torch.sigmoid(((x[:, t, self.input_dim_s:] @ self.u_r_l) + (h_tilda_t_l @ self.w_r_l) + self.b_r_l))
            z_t_l = torch.sigmoid(((x[:, t, self.input_dim_s:] @ self.u_z_l) + (h_tilda_t_l @ self.w_z_l) + self.b_z_l))
            g_t_l = torch.tanh(
                ((x[:, t, self.input_dim_s:] @ self.u_h_l) + ((h_tilda_t_l * r_t_l) @ self.w_h_l) + self.b_h_l))

            h_tilda_t_l = z_t_l * h_tilda_t_l + (1 - z_t_l) * g_t_l

            if(t>=self.short):

                weight_s= _estimate_alpha1(H_tilda_t_s, targets=h_tilda_t_s) #shape batch, feature 1,

                weight_l = _estimate_alpha1(H_tilda_t_l, targets=h_tilda_t_l)  # shape batch, feature 1,

                #近似值
                h_tilda_t_s=torch.bmm(H_tilda_t_s.permute(0,2,1),weight_s).squeeze(-1)

                h_tilda_t_l = torch.bmm(H_tilda_t_l.permute(0, 2, 1), weight_l).squeeze(-1)

                unorm_weight_s=weight_s
                unorm_weight_l = weight_l

                newhtilda_t=torch.concat([h_tilda_t_s,h_tilda_t_l],dim=1)

                pred_y = (self.classify(newhtilda_t)) #shape  batch, 2

                #shop weights, loan weights
                all_weights=self.classify.weight.permute(1,0)
                all_weights_s=torch.norm(all_weights[:self.n_units,:]).unsqueeze(0)
                all_weights_l=torch.norm(all_weights[self.n_units:,:]).unsqueeze(0)
                all_weights=torch.concat([all_weights_s,all_weights_l])
                # 保存每次的预测值
                loss_CE_per_time = torch.nn.CrossEntropyLoss(weight=weights)(pred_y, target[:, t - 1].long())
                loss_CE = loss_CE + loss_CE_per_time  # 整个序列的loss的均值

                # 保存每次的预测值
                softmax = torch.nn.Softmax(dim=1)
                pred_y = softmax(pred_y)
                prediction = torch.argmax(pred_y, dim=1)  # shape batch
                prob_list += [pred_y[:, 1]]  # 这里为了计算auc，保存类别为1的概率值
                pred_list += [prediction]

                unorm_list_s += [unorm_weight_s]
                unorm_list_l += [unorm_weight_l]
                weights_list+=[all_weights]

        pred = torch.stack(pred_list).permute(1,0)
        unorm_s = torch.stack(unorm_list_s) #shape time_depth, BATCH input_dim
        unorm_l = torch.stack(unorm_list_l)
        weights=torch.stack(weights_list)

        return loss_CE, pred,None, unorm_s, unorm_l,weights,None

class shop_two_Delegru(torch.jit.ScriptModule):
    def __init__(self, Delegru_params, short):
        super(shop_two_Delegru, self).__init__()
        self.input_dim_s=Delegru_params['input_dim_s']
        self.input_dim_l = Delegru_params['input_dim_l']

        self.n_units=Delegru_params['n_units']
        self.depth = Delegru_params['time_depth']
        self.output_dim = Delegru_params['output_dim']
        self.N_units = Delegru_params['N_units']
        self.bias = Delegru_params['bias']
        self.short=short

        self.U_r_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.U_z_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.U_h_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.W_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.W_z_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.W_h_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.B_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units))
        self.B_z_s = nn.Parameter(torch.Tensor(self.input_dim_s, self.n_units))
        self.B_h_s = nn.Parameter(torch.Tensor(self.input_dim_s, self.n_units))

        # shop vanilla LSTM parameter
        self.u_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units))  # 正态分布
        self.w_r_s = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.b_r_s = nn.Parameter(torch.zeros(self.N_units))
        self.w_z_s = nn.Parameter(torch.randn(self.N_units, self.N_units))  # 输入维度，输出维度
        self.u_z_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units))
        self.b_z_s = nn.Parameter(torch.zeros(self.N_units))
        self.w_h_s = nn.Parameter(torch.randn(self.N_units, self.N_units))  # 输入维度，输出维度
        self.u_h_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units))
        self.b_h_s = nn.Parameter(torch.zeros(self.N_units))

        self.w_p_s = nn.Parameter(torch.randn(self.N_units, self.output_dim))
        self.b_p_s = nn.Parameter(torch.zeros(self.output_dim))

        self.classify = torch.nn.Linear(self.N_units, self.output_dim, bias=self.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B_f' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'b_f' in name:
                nn.init.constant_(param.data, 0.01)

            elif 'B' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                #nn.init.orthogonal_(param.data)
                nn.init.xavier_uniform_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                #nn.init.orthogonal_(param.data)
                nn.init.xavier_uniform_(param.data)



    def forward(self, x,target, device):
        #x=torch.cuda.DoubleTensor(x)
        #每个时刻每个feature的hidden， W,U,B

        H_tilda_t_s = (torch.zeros(x.shape[0], self.input_dim_s, self.n_units)).to(device)

        # 每个时刻对应所有feature的hidden
        h_tilda_t_s = torch.zeros(x.shape[0],  self.N_units).to(device)

        unorm_list_s = torch.jit.annotate(list[Tensor], [])

        pred_list = torch.jit.annotate(list[Tensor], [])
        prob_list = torch.jit.annotate(list[Tensor], [])
        #weights=torch.FloatTensor([0.2,0.8]).cuda(1)
        weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)

        loss_CE = torch.as_tensor(0.0)
        #print('new begin')
        for t in range(self.depth-1):

            # eq 1 先算每个feature的hidden，在算所有feature对应的hidden
            #先算shop
            R_tilda_t_s = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_s, self.W_r_s) + \
                                        torch.einsum("bij,jik->bjk", x[:, t, :self.input_dim_s].unsqueeze(1),
                                                     self.U_r_s) + self.B_r_s)
            Z_tilda_t_s = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_s, self.W_z_s) + \
                                        torch.einsum("bij,jik->bjk", x[:, t, :self.input_dim_s].unsqueeze(1),
                                                     self.U_z_s) + self.B_z_s)
            G_tilda_t_s = torch.tanh(torch.einsum("bij,ijk->bik", (H_tilda_t_s * R_tilda_t_s), self.W_h_s) + \
                                     torch.einsum("bij,jik->bjk", x[:, t, :self.input_dim_s].unsqueeze(1),
                                                  self.U_h_s) + self.B_h_s)

            H_tilda_t_s = Z_tilda_t_s * H_tilda_t_s + (1 - Z_tilda_t_s) * G_tilda_t_s  # shape batch, feature, dim

            # 所有lstm的关系 shop
            r_t_s = torch.sigmoid(((x[:, t, :self.input_dim_s] @ self.u_r_s) + (h_tilda_t_s @ self.w_r_s) + self.b_r_s))
            z_t_s = torch.sigmoid(((x[:, t, :self.input_dim_s] @ self.u_z_s) + (h_tilda_t_s @ self.w_z_s) + self.b_z_s))
            g_t_s = torch.tanh(
                ((x[:, t, :self.input_dim_s] @ self.u_h_s) + ((h_tilda_t_s * r_t_s) @ self.w_h_s) + self.b_h_s))

            h_tilda_t_s = z_t_s * h_tilda_t_s + (1 - z_t_s) * g_t_s

            #print('htilda_t nan:{}'.format(torch.isnan(h_tilda_t).int().sum()))

            if(t>=self.short):

                weight_s= _estimate_alpha1(H_tilda_t_s, targets=h_tilda_t_s) #shape batch, feature 1,
                #近似值
                h_tilda_t_s=torch.bmm(H_tilda_t_s.permute(0,2,1),weight_s).squeeze(-1)

                unorm_weight_s=weight_s

                newhtilda_t=h_tilda_t_s

                pred_y = (self.classify(newhtilda_t)) #shape  batch, 2


                loss_CE_per_time = torch.nn.CrossEntropyLoss(weight=weights)(pred_y.float(), target[:,t-1].long())
                #print('each time loss:{}'.format(loss_CE_per_time))
                #print('pred_y nan:{}'.format(torch.isnan(pred_y).int().sum()))
                #print(loss_CE_per_time)#每次预测，loss的均值
                #loss_CE_per_time = torch.nn.CrossEntropyLoss()(pred_y,target[:, t - 1].long())  # 每次预测，loss的均值
                loss_CE = loss_CE + loss_CE_per_time #整个序列的loss的均值

                #保存每次的预测值
                softmax = torch.nn.Softmax(dim=1)
                pred_y = softmax(pred_y)
                prediction = torch.argmax(pred_y, dim=1) #shape batch
                #prediction=torch.round(pred_y)
                prob_list+=[pred_y[:,1]] #这里为了计算auc，保存类别为1的概率值
                #prob_list += [pred_y]
                pred_list+=[prediction]
                unorm_list_s += [unorm_weight_s]

        pred = torch.stack(pred_list).permute(1,0)
       # prob = torch.stack(prob_list).permute(1,0)
        unorm_s = torch.stack(unorm_list_s) #shape time_depth, BATCH input_dim


        return loss_CE, pred,None, unorm_s, None,None,None

class loan_two_Delegru(torch.jit.ScriptModule):
    def __init__(self, Delegru_params, short):
        super(loan_two_Delegru, self).__init__()
        self.input_dim_s=Delegru_params['input_dim_s']
        self.input_dim_l = Delegru_params['input_dim_l']

        self.n_units=Delegru_params['n_units']
        self.depth = Delegru_params['time_depth']
        self.output_dim = Delegru_params['output_dim']
        self.N_units = Delegru_params['N_units']
        self.bias = Delegru_params['bias']
        self.short=short

        #loan tensor LSTM parameter
        self.U_r_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.U_z_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.U_h_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.W_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.W_z_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.W_h_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.B_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units))
        self.B_z_l = nn.Parameter(torch.Tensor(self.input_dim_l, self.n_units))
        self.B_h_l = nn.Parameter(torch.Tensor(self.input_dim_l, self.n_units))

        #loan vanilla LSTM parameter

        self.u_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )  # 正态分布
        self.w_r_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.b_r_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_z_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_z_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )
        self.b_z_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_h_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_h_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )
        self.b_h_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_p_l = nn.Parameter(torch.randn(self.N_units, self.output_dim))
        self.b_p_l = nn.Parameter(torch.zeros(self.output_dim))

        self.classify = torch.nn.Linear(self.N_units, self.output_dim, bias=self.bias)


        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B_f' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'b_f' in name:
                nn.init.constant_(param.data, 0.01)

            elif 'B' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                #nn.init.orthogonal_(param.data)
                nn.init.xavier_uniform_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                #nn.init.orthogonal_(param.data)
                nn.init.xavier_uniform_(param.data)



    def forward(self, x,target, device):
        #x=torch.cuda.DoubleTensor(x)
        #每个时刻每个feature的hidden， W,U,B

        H_tilda_t_l = (torch.zeros(x.shape[0], self.input_dim_l, self.n_units)).to(device)

        h_tilda_t_l = torch.zeros(x.shape[0], self.N_units).to(device)

        unorm_list_l = torch.jit.annotate(list[Tensor], [])

        pred_list = torch.jit.annotate(list[Tensor], [])
        prob_list = torch.jit.annotate(list[Tensor], [])
        #weights=torch.FloatTensor([0.2,0.8]).cuda(1)
        weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)

        loss_CE = torch.as_tensor(0.0)
        #print('new begin')
        for t in range(self.depth-1):

            R_tilda_t_l = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_l, self.W_r_l) + \
                                        torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                     self.U_r_l) + self.B_r_l)
            Z_tilda_t_l = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_l, self.W_z_l) + \
                                        torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                     self.U_z_l) + self.B_z_l)
            G_tilda_t_l = torch.tanh(torch.einsum("bij,ijk->bik", (H_tilda_t_l * R_tilda_t_l), self.W_h_l) + \
                                     torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                  self.U_h_l) + self.B_h_l)

            H_tilda_t_l = Z_tilda_t_l * H_tilda_t_l + (1 - Z_tilda_t_l) * G_tilda_t_l  # shape batch, feature, dim

            # 所有lstm的关系 shop
            r_t_l = torch.sigmoid(((x[:, t, self.input_dim_s:] @ self.u_r_l) + (h_tilda_t_l @ self.w_r_l) + self.b_r_l))
            z_t_l = torch.sigmoid(((x[:, t, self.input_dim_s:] @ self.u_z_l) + (h_tilda_t_l @ self.w_z_l) + self.b_z_l))
            g_t_l = torch.tanh(
                ((x[:, t, self.input_dim_s:] @ self.u_h_l) + ((h_tilda_t_l * r_t_l) @ self.w_h_l) + self.b_h_l))

            h_tilda_t_l = z_t_l * h_tilda_t_l + (1 - z_t_l) * g_t_l

            if(t>=self.short):

                weight_l = _estimate_alpha1(H_tilda_t_l, targets=h_tilda_t_l)  # shape batch, feature 1,

                #近似值
                h_tilda_t_l = torch.bmm(H_tilda_t_l.permute(0, 2, 1), weight_l).squeeze(-1)

                unorm_weight_l = weight_l

                newhtilda_t=h_tilda_t_l

                pred_y = (self.classify(newhtilda_t)) #shape  batch, 2

                loss_CE_per_time = torch.nn.CrossEntropyLoss(weight=weights)(pred_y.float(), target[:,t-1].long())
                #print('each time loss:{}'.format(loss_CE_per_time))
                #print('pred_y nan:{}'.format(torch.isnan(pred_y).int().sum()))
                #print(loss_CE_per_time)#每次预测，loss的均值
                #loss_CE_per_time = torch.nn.CrossEntropyLoss()(pred_y,target[:, t - 1].long())  # 每次预测，loss的均值
                loss_CE = loss_CE + loss_CE_per_time #整个序列的loss的均值

                #保存每次的预测值
                softmax = torch.nn.Softmax(dim=1)
                pred_y = softmax(pred_y)
                prediction = torch.argmax(pred_y, dim=1) #shape batch
                #prediction=torch.round(pred_y)
                prob_list+=[pred_y[:,1]] #这里为了计算auc，保存类别为1的概率值
                #prob_list += [pred_y]
                pred_list+=[prediction]

                unorm_list_l += [unorm_weight_l]


        pred = torch.stack(pred_list).permute(1,0)
        prob = torch.stack(prob_list).permute(1,0)

        unorm_l = torch.stack(unorm_list_l)


        return loss_CE, pred,None, None, unorm_l,None,None

########################   Baseline model      ######################################
#IMV-Tensor LSTM ####
class IMVTensorLSTM_pertime(torch.jit.ScriptModule):
    def __init__(self, IMVTensorLSTM_pertime_params, short):
        super(IMVTensorLSTM_pertime, self).__init__()
        self.input_dim = IMVTensorLSTM_pertime_params['input_dim']
        self.n_units = IMVTensorLSTM_pertime_params['n_units']
        self.depth = IMVTensorLSTM_pertime_params['time_depth']
        self.output_dim = IMVTensorLSTM_pertime_params['output_dim']
        self.bias= IMVTensorLSTM_pertime_params['bias']

        self.U_j = nn.Parameter(torch.randn( self.input_dim, 1,  self.n_units) )
        self.U_i = nn.Parameter(torch.randn( self.input_dim, 1,  self.n_units) )
        self.U_f = nn.Parameter(torch.randn( self.input_dim, 1,  self.n_units) )
        self.U_o = nn.Parameter(torch.randn( self.input_dim, 1,  self.n_units) )
        self.W_j = nn.Parameter(torch.randn( self.input_dim,  self.n_units,  self.n_units) )
        self.W_i = nn.Parameter(torch.randn( self.input_dim,  self.n_units,  self.n_units) )
        self.W_f = nn.Parameter(torch.randn( self.input_dim,  self.n_units,  self.n_units) )
        self.W_o = nn.Parameter(torch.randn( self.input_dim,  self.n_units,  self.n_units) )
        self.b_j = nn.Parameter(torch.randn( self.input_dim,  self.n_units) )
        self.b_i = nn.Parameter(torch.randn( self.input_dim,  self.n_units) )
        self.b_f = nn.Parameter(torch.randn( self.input_dim,  self.n_units) )
        self.b_o = nn.Parameter(torch.randn( self.input_dim,  self.n_units) )
        self.F_alpha_n = nn.Parameter(torch.randn( self.input_dim,  self.n_units, 1) )
        self.F_alpha_n_b = nn.Parameter(torch.randn( self.input_dim, 1) )
        self.F_beta = nn.Linear(2 *  self.n_units, 1)
        #self.Phi = nn.Linear(2 *  self.n_units, self.output_dim)
        self.short=short
        self.reset_parameters()

        self.classify = nn.Sequential(
            # torch.nn.Linear(self.n_units*2, self.n_units, bias=self.bias),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.n_units*2, self.output_dim, bias=self.bias),

        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():

            if 'b' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)

    #@torch.jit.script_method
    def forward(self, x, target, device,train=True):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        outputs = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])
        prob_list = torch.jit.annotate(list[Tensor], [])
        beta_list = torch.jit.annotate(list[Tensor], [])
        alpha_list = torch.jit.annotate(list[Tensor], [])
        #weights = torch.FloatTensor([0.2, 0.8]).cuda(1)
        weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)

        #weights = torch.FloatTensor([1.0637, 16.6962]).cuda(1)

        loss_CE = torch.as_tensor(0.0)
        for t in range(self.depth-1):
            outputs += [h_tilda_t]
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t * torch.tanh(c_tilda_t))
            if (t >= self.short):
                newoutputs = torch.stack(outputs)
                newoutputs = newoutputs.permute(1, 0, 2, 3)
            # eq 8
                alphas = torch.tanh(torch.einsum("btij,ijk->btik", newoutputs, self.F_alpha_n) + self.F_alpha_n_b)
                alphas = torch.exp(alphas)
                alphas = alphas / torch.sum(alphas, dim=1, keepdim=True) #compute the contribuiton of past time point for each variable
                g_n = torch.sum(alphas * newoutputs, dim=1)
                hg = torch.cat([g_n, h_tilda_t], dim=2)
                mu = self.classify(hg) #the classification results of each variable
                betas = torch.tanh(self.F_beta(hg))
                betas = torch.exp(betas)
                betas = betas / torch.sum(betas, dim=1, keepdim=True)
                mean = torch.sum(betas * mu, dim=1)

                alpha_list+=[alphas]
                beta_list+=[betas]
                loss_CE_per_time = torch.nn.CrossEntropyLoss(weight=weights)(mean,
                                                                         target[:, t - 1].long())  # 每次预测，loss的均值
                loss_CE = loss_CE + loss_CE_per_time  # 整个序列的loss的均值

            # 保存每次的预测值
                softmax = torch.nn.Softmax(dim=1)
                pred_y = softmax(mean)
                prediction = torch.argmax(pred_y, dim=1)  # shape batch
            # prediction=torch.round(pred_y)
                #prob_list += [mean[:, 1]]
                pred_list += [prediction]

        pred = torch.stack(pred_list).permute(1, 0)
        #prob = torch.stack(prob_list).permute(1, 0)
        alpha_list=torch.cat(alpha_list, dim=1).squeeze(-1)
        beta_list=torch.stack(beta_list).squeeze(-1).permute(1,0,2)

        return loss_CE,pred,None,alpha_list,beta_list, None,None
    #pred shape 128,23
    #prob shape 128,23

###RETAIN LSTM ####
class Retain_pertime(nn.Module):
    def __init__(self, Retain_pertime_params, short):
        super(Retain_pertime, self).__init__()
        self.inputDimSize = Retain_pertime_params['inputDimSize']
        self.embDimSize =Retain_pertime_params['embDimSize']
        self.alphaHiddenDimSize = Retain_pertime_params['alphaHiddenDimSize']
        self.betaHiddenDimSize = Retain_pertime_params['betaHiddenDimSize']
        self.outputDimSize = Retain_pertime_params['output_dim']
        self.keep_prob = Retain_pertime_params['keep_prob']
        self.bias=Retain_pertime_params['bias']
        self.embedding = nn.Linear(self.inputDimSize, self.embDimSize)
        self.dropout = nn.Dropout(self.keep_prob)
        self.gru_alpha = nn.GRU(self.embDimSize, self.alphaHiddenDimSize)
        self.gru_beta = nn.GRU(self.embDimSize, self.betaHiddenDimSize)
        self.alpha_att = nn.Linear(self.alphaHiddenDimSize, 1)
        self.beta_att = nn.Linear(self.betaHiddenDimSize, self.embDimSize)
        self.out = nn.Linear(self.embDimSize, self.outputDimSize)
        self.short=short
        self.classify = nn.Sequential(
            # torch.nn.Linear(self.embDimSize, self.embDimSize, bias=self.bias),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.embDimSize, self.outputDimSize, bias=self.bias)
        )

    def initHidden_alpha(self, batch_size,device):
        return torch.zeros(1, batch_size, self.alphaHiddenDimSize).to(device)

    def initHidden_beta(self, batch_size, device):
        return torch.zeros(1, batch_size, self.betaHiddenDimSize).to(device)

    def attentionStep(self, h_a, h_b, att_timesteps):
        reverse_emb_t = self.emb[:att_timesteps].flip(dims=[0]) #index 0:time-1
        reverse_h_a = self.gru_alpha(reverse_emb_t, h_a)[0].flip(dims=[0]) * 0.5
        reverse_h_b = self.gru_beta(reverse_emb_t, h_b)[0].flip(dims=[0]) * 0.5

        preAlpha = self.alpha_att(reverse_h_a)
        preAlpha = torch.squeeze(preAlpha, dim=2)
        alpha = torch.transpose(F.softmax(torch.transpose(preAlpha, 0, 1)), 0, 1)
        beta = torch.tanh(self.beta_att(reverse_h_b))

        c_t = torch.sum((alpha.unsqueeze(2) * beta * self.emb[:att_timesteps]), dim=0)
        return c_t, alpha, beta

    def forward(self, x, target, device):
        temp=x.permute(1,0,2)
        first_h_a = self.initHidden_alpha(temp.shape[1], device)
        first_h_b = self.initHidden_beta(temp.shape[1], device)

        self.emb = self.embedding(temp)  #shape depth, batch, embedding dim
        w_emb=self.embedding.weight.data
        if self.keep_prob < 1:
            self.emb = self.dropout(self.emb)

        count = np.arange(temp.shape[0]-1)+1 #表示时刻数
        weight_list = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])
        prob_list = torch.jit.annotate(list[Tensor], [])
        #weights = torch.FloatTensor([0.2, 0.8]).cuda(1)
        #weights = torch.FloatTensor([1.0637, 16.6962]).cuda(1)
        weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
        loss_CE = torch.as_tensor(0.0)
        for i, att_timesteps in enumerate(count): # i from 0 to depth-1, att_timsteps from 1 to depth

             c_t, alpha, beta = self.attentionStep(first_h_a, first_h_b, att_timesteps)#alpha t, batch
             #beta time, batch, embsize, c_t shape batch, embsize
             y_hat=self.out(c_t)
             w_out=self.out.weight.data #shape 2, 32
             if(i>=0):

             #compute variable importance for each time prediction
                new_beta = beta.permute(1, 0, 2).unsqueeze(-1)
                d = torch.mul(new_beta, w_emb)
                e =torch.sum( torch.matmul(w_out, d) .squeeze(2),dim=2)
                #new_alpha = alpha.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
                new_alpha = alpha.permute(1, 0).unsqueeze(-1)
                f = torch.mul(new_alpha, e)
                f=torch.mul(f,x[:,:att_timesteps,:])
                g = torch.mean(f, dim=1)
                weight_list +=[g]
                softmax = torch.nn.Softmax(dim=1)
                pred_y = softmax(y_hat)
                prediction = torch.argmax(pred_y, dim=1)  # shape batch
             # prediction=torch.round(pred_y)
                #prob_list += [y_hat[:, 1]]
                pred_list += [prediction]
                loss_CE_per_time = torch.nn.CrossEntropyLoss(weight=weights)(y_hat,
                                                                          target[:, i - 1].long())  # 每次预测，loss的均值
                loss_CE = loss_CE + loss_CE_per_time  # 整个序列的loss的均值



        weight_list=torch.stack(weight_list).permute(1,0,2)
        pred = torch.stack(pred_list).permute(1, 0)
        #prob = torch.stack(prob_list).permute(1, 0)
        return loss_CE, pred,None,weight_list,None,None,None

###LR##
class normalLRpertime(torch.jit.ScriptModule):
    def __init__(self,normalLRpertime_params, short):
        super(normalLRpertime, self).__init__()
        self.model_logistic = LogisticRegression \
            ( solver='sag', multi_class='multinomial', max_iter=500)
        self.short=short
        self.depth = normalLRpertime_params['time_depth']

    def forward(self, x_train, target_train,x_test, target_test, device):
        pred_list = torch.jit.annotate(list[Tensor], [])
        for t in range(self.depth-1):
                train_shape = x_train[:, :(t + 1), :].shape[0]
                new_x_train = x_train[:, :t + 1, :].reshape(train_shape, -1)
                model = self.model_logistic.fit(new_x_train, target_train[:, t - 1])
                test_shape = x_test[:, :t + 1, :].shape[0]
                new_x_test = x_test[:, :t + 1, :].reshape(test_shape, -1)
                pred_y = model.predict(new_x_test)
                # 保存每次的预测值
                pred_list += [pred_y]

        pred=np.array(pred_list).transpose(1,0)
        return None,pred,None,None,None, None,None

###SVM
class normalSVMpertime(torch.jit.ScriptModule):
    def __init__(self,normalSVMpertime_params, short):
        super(normalSVMpertime, self).__init__()
        self.model_svm = svm.SVC(kernel='linear', decision_function_shape='ovo')
        self.short=short
        self.depth = normalSVMpertime_params['time_depth']

    def forward(self, x_train, target_train,x_test, target_test, device):
        pred_list = torch.jit.annotate(list[Tensor], [])
        for t in range(self.depth-1):
                train_shape = x_train[:, :(t + 1), :].shape[0]
                new_x_train = x_train[:, :t + 1, :].reshape(train_shape, -1)
                model = self.model_svm.fit(new_x_train, target_train[:, t - 1])

                test_shape = x_test[:, :t + 1, :].shape[0]
                new_x_test = x_test[:, :t + 1, :].reshape(test_shape, -1)
                pred_y = model.predict(new_x_test)
                pred_list += [pred_y]

        pred=np.array(pred_list).transpose(1,0)

        return None,pred, None,None,None, None,None

###RF
class normalRFpertime(torch.jit.ScriptModule):
    def __init__(self,normalRFpertime_params, short):
        super(normalRFpertime, self).__init__()
        self.model_random_forest_classifier = RandomForestClassifier(n_estimators=5,criterion='gini')
        self.short=short
        self.depth = normalRFpertime_params['time_depth']

    def forward(self, x_train, target_train,x_test, target_test, device):
        pred_list = torch.jit.annotate(list[Tensor], [])

        for t in range(self.depth-1):
                train_shape=x_train[:, :(t+1), :].shape[0]
                new_x_train=x_train[:, :t+1, :].reshape(train_shape,-1)
                model=self.model_random_forest_classifier.fit(new_x_train, target_train[:,t-1])

                test_shape = x_test[:, :t+1, :].shape[0]
                new_x_test = x_test[:, :t+1, :].reshape(test_shape, -1)
                pred_y = model.predict(new_x_test)
                pred_list += [pred_y]

        pred=np.array(pred_list).transpose(1,0)

        return None,pred, None,None,None, None,None

###MLP
class normalMLPpertime(torch.jit.ScriptModule):
    def __init__(self,normalMLPpertime_params, short):
        super(normalMLPpertime, self).__init__()
        self.output_dim=normalMLPpertime_params['output_dim']
        self.layer_size=normalMLPpertime_params['layer_size']
        self.model_mlp_classifier = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(self.layer_size,self.output_dim))
        self.short=short
        self.depth = normalMLPpertime_params['time_depth']

    def forward(self, x_train, target_train,x_test, target_test, device):

        pred_list = torch.jit.annotate(list[Tensor], [])

        for t in range(self.depth-1):
                train_shape=x_train[:, :(t+1), :].shape[0]
                new_x_train=x_train[:, :t+1, :].reshape(train_shape,-1)
                model=self.model_mlp_classifier.fit(new_x_train, target_train[:,t-1])

                test_shape = x_test[:, :t+1, :].shape[0]
                new_x_test = x_test[:, :t+1, :].reshape(test_shape, -1)
                pred_y = model.predict(new_x_test)

                # 保存每次的预测值
                pred_list += [pred_y]

        pred=np.array(pred_list).transpose(1,0)

        return None,pred,None,None,None, None,None

###LDA
class normalLDApertime(torch.jit.ScriptModule):
    def __init__(self,normalLDApertime_params, short):
        super(normalLDApertime, self).__init__()
        self.model_lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
        self.short=short
        self.depth = normalLDApertime_params['time_depth']

    def forward(self, x_train, target_train,x_test, target_test, device):
        pred_list = torch.jit.annotate(list[Tensor], [])
        for t in range(self.depth-1):
                train_shape = x_train[:, :(t + 1), :].shape[0]
                new_x_train = x_train[:, :t + 1, :].reshape(train_shape, -1)
                model = self.model_lda.fit(new_x_train, target_train[:, t - 1])

                test_shape = x_test[:, :t + 1, :].shape[0]
                new_x_test = x_test[:, :t + 1, :].reshape(test_shape, -1)
                pred_y = model.predict(new_x_test)
                # 保存每次的预测值
                pred_list += [pred_y]
        pred=np.array(pred_list).transpose(1,0)
        return None,pred,None,None,None, None,None

###Naive Bayes
class normalNBpertime(torch.jit.ScriptModule):
    def __init__(self,normalNBpertime_params, short):
        super(normalNBpertime, self).__init__()
        self.model_nb = MultinomialNB()
        self.short=short
        self.depth = normalNBpertime_params['time_depth']

    def forward(self, x_train, target_train,x_test, target_test, device):
        pred_list = torch.jit.annotate(list[Tensor], [])
        for t in range(self.depth-1):
                train_shape = x_train[:, :(t + 1), :].shape[0]
                new_x_train = x_train[:, :t + 1, :].reshape(train_shape, -1)
                model = self.model_nb.fit(new_x_train, target_train[:, t - 1])

                test_shape = x_test[:, :t + 1, :].shape[0]
                new_x_test = x_test[:, :t + 1, :].reshape(test_shape, -1)
                pred_y = model.predict(new_x_test)
                # 保存每次的预测值
                pred_list += [pred_y]

        pred=np.array(pred_list).transpose(1,0)
        return None,pred,None,None,None, None,None

###LSTM
class normalLSTMpertime(torch.jit.ScriptModule):
    def __init__(self, normalLSTMpertime_params, short):
        super(normalLSTMpertime, self).__init__()
        self.input_dim=normalLSTMpertime_params['input_dim']
        self.n_units=normalLSTMpertime_params['n_units']
        self.depth = normalLSTMpertime_params['time_depth']
        self.output_dim = normalLSTMpertime_params['output_dim']
        self.bias = normalLSTMpertime_params['bias']
        self.num_classes = normalLSTMpertime_params['num_classes']

        self.U_j = nn.Parameter(torch.randn(self.input_dim, self.n_units)  ) #正态分布
        self.W_j = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.b_j = nn.Parameter(torch.zeros(self.n_units))
        self.W_i = nn.Parameter(torch.randn(self.n_units, self.n_units)  ) #输入维度，输出维度
        self.U_i=nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_i = nn.Parameter(torch.zeros(self.n_units))
        self.W_f = nn.Parameter(torch.randn(self.n_units, self.n_units)  )  # 输入维度，输出维度
        self.U_f = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_f = nn.Parameter(torch.zeros(self.n_units))
        self.W_o = nn.Parameter(torch.randn(self.n_units, self.n_units)  )  # 输入维度，输出维度
        self.U_o = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_o = nn.Parameter(torch.zeros(self.n_units))
        self.W_p=nn.Parameter(torch.randn(self.n_units, self.output_dim))
        self.b_p= nn.Parameter(torch.zeros(self.output_dim))
        self.short = short


        self.classify = nn.Sequential(
            torch.nn.Linear(self.n_units, self.output_dim, bias=self.bias),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
            for name, param in self.named_parameters():
                if 'B' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'U' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'W' in name:
                    nn.init.orthogonal_(param.data)
                elif 'b' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'u' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'w' in name:
                    nn.init.orthogonal_(param.data)

    def forward(self, x, target, device):
        h_tilda_t = torch.zeros(x.shape[0], self.n_units).to(device)
        c_tilda_t = torch.zeros(x.shape[0], self.n_units).to(device)
        pred_list = torch.jit.annotate(list[Tensor], [])
        weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
        loss_CE = torch.as_tensor(0.0)

        for t in range(self.depth-1):
            i_t=torch.sigmoid(((x[:,t,:]@self.U_i)+(h_tilda_t@self.W_i)+self.b_i))
            f_t = torch.sigmoid(((x[:, t, :] @ self.U_f) + (h_tilda_t @ self.W_f) + self.b_f))
            o_t = torch.sigmoid(((x[:, t, :] @ self.U_o) + (h_tilda_t @ self.W_o) + self.b_o))
            j_t=torch.tanh(((x[:,t,:]@self.U_j)+(h_tilda_t@self.W_j)+self.b_j))
            c_tilda_t=f_t*c_tilda_t+i_t*j_t
            h_tilda_t=o_t*torch.tanh(c_tilda_t)

            pred_y = (self.classify(h_tilda_t))
            loss_CE_per_time = torch.nn.CrossEntropyLoss(weight=weights)(pred_y, target[:, t - 1].long())
            loss_CE = loss_CE + loss_CE_per_time  # 整个序列的loss的均值

            # 保存每次的预测值
            softmax = torch.nn.Softmax(dim=1)
            pred_y = softmax(pred_y)
            prediction = torch.argmax(pred_y, dim=1)  # shape batch
            pred_list += [prediction]

        pred = torch.stack(pred_list).permute(1, 0)
        return loss_CE,pred,None,None,None, None,None

###GRU
class normalGRUpertime(torch.jit.ScriptModule):
    def __init__(self, normalGRUpertime_params, short):
        super(normalGRUpertime, self).__init__()
        self.input_dim=normalGRUpertime_params['input_dim']
        self.n_units=normalGRUpertime_params['n_units']
        self.depth = normalGRUpertime_params['time_depth']
        self.output_dim = normalGRUpertime_params['output_dim']
        self.bias = normalGRUpertime_params['bias']
        self.num_classes = normalGRUpertime_params['num_classes']

        self.U_r = nn.Parameter(torch.randn(self.input_dim, self.n_units)  ) #正态分布
        self.W_r = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.b_r = nn.Parameter(torch.zeros(self.n_units))
        self.W_z = nn.Parameter(torch.randn(self.n_units, self.n_units)  ) #输入维度，输出维度
        self.U_z=nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_z = nn.Parameter(torch.zeros(self.n_units))
        self.W_h = nn.Parameter(torch.randn(self.n_units, self.n_units)  )  # 输入维度，输出维度
        self.U_h = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_h = nn.Parameter(torch.zeros(self.n_units))
        self.W_p=nn.Parameter(torch.randn(self.n_units, self.output_dim))
        self.b_p= nn.Parameter(torch.zeros(self.output_dim))
        self.short = short
        self.classify = nn.Sequential(
            torch.nn.Linear(self.n_units, self.output_dim, bias=self.bias),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
            for name, param in self.named_parameters():
                if 'B' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'U' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'W' in name:
                    nn.init.orthogonal_(param.data)
                elif 'b' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'u' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'w' in name:
                    nn.init.orthogonal_(param.data)

    def forward(self, x, target, device):
        h_tilda_t = torch.zeros(x.shape[0], self.n_units).to(device)
        pred_list = torch.jit.annotate(list[Tensor], [])
        weights = torch.FloatTensor([3.44,5.52,1.89]).cuda(1)
        loss_CE = torch.as_tensor(0.0)

        for t in range(self.depth-1):
            r_t=torch.sigmoid(((x[:,t,:]@self.U_r)+(h_tilda_t@self.W_r)+self.b_r))
            z_t = torch.sigmoid(((x[:, t, :] @ self.U_z) + (h_tilda_t @ self.W_z) + self.b_z))
            g_t = torch.tanh(((x[:, t, :] @ self.U_h) + (h_tilda_t * r_t)@self.W_h + self.b_h))
            h_tilda_t=z_t*h_tilda_t+(1-z_t)*g_t
            pred_y = (self.classify(h_tilda_t))
            loss_CE_per_time = torch.nn.CrossEntropyLoss(weight=weights)(pred_y, target[:, t - 1].long())

            # 保存每次的预测值
            softmax = torch.nn.Softmax(dim=1)
            pred_y = softmax(pred_y)
            prediction = torch.argmax(pred_y, dim=1)  # shape batch
            pred_list += [prediction]
            loss_CE = loss_CE + loss_CE_per_time  # 整个序列的loss的均值

        pred = torch.stack(pred_list).permute(1, 0)

        return loss_CE,pred, None,None,None, None,None


class evaluate_normalGRUpertime(torch.jit.ScriptModule):
    def __init__(self, normalGRUpertime_params, short):
        super(evaluate_normalGRUpertime, self).__init__()
        self.input_dim=normalGRUpertime_params['input_dim']
        self.n_units=normalGRUpertime_params['n_units']
        self.depth = normalGRUpertime_params['time_depth']
        self.output_dim = normalGRUpertime_params['output_dim']
        self.bias = normalGRUpertime_params['bias']
        self.num_classes = normalGRUpertime_params['num_classes']

        self.U_r = nn.Parameter(torch.randn(self.input_dim, self.n_units)  ) #正态分布
        self.W_r = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.b_r = nn.Parameter(torch.zeros(self.n_units))
        self.W_z = nn.Parameter(torch.randn(self.n_units, self.n_units)  ) #输入维度，输出维度
        self.U_z=nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_z = nn.Parameter(torch.zeros(self.n_units))
        self.W_h = nn.Parameter(torch.randn(self.n_units, self.n_units)  )  # 输入维度，输出维度
        self.U_h = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_h = nn.Parameter(torch.zeros(self.n_units))
        self.W_p=nn.Parameter(torch.randn(self.n_units, self.output_dim))
        self.b_p= nn.Parameter(torch.zeros(self.output_dim))
        self.short = short
        self.classify = nn.Sequential(
            torch.nn.Linear(self.n_units, self.output_dim, bias=self.bias),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
            for name, param in self.named_parameters():
                if 'B' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'U' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'W' in name:
                    nn.init.orthogonal_(param.data)
                elif 'b' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'u' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'w' in name:
                    nn.init.orthogonal_(param.data)

    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.n_units)
        pred_list = torch.jit.annotate(list[Tensor], [])
        prob_list = torch.jit.annotate(list[Tensor], [])

        for t in range(self.depth-1):
            r_t=torch.sigmoid(((x[:,t,:]@self.U_r)+(h_tilda_t@self.W_r)+self.b_r))
            z_t = torch.sigmoid(((x[:, t, :] @ self.U_z) + (h_tilda_t @ self.W_z) + self.b_z))
            g_t = torch.tanh(((x[:, t, :] @ self.U_h) + (h_tilda_t * r_t)@self.W_h + self.b_h))
            h_tilda_t=z_t*h_tilda_t+(1-z_t)*g_t

            if (t > self.short):
                pred_y = (self.classify(h_tilda_t))

                # 保存每次的预测值
                softmax = torch.nn.Softmax(dim=1)
                pred_y = softmax(pred_y)
                prob_list += [pred_y]

                #prediction = torch.argmax(pred_y, dim=1)  # shape batch
                #pred_list += [prediction]
                #loss_CE = loss_CE + loss_CE_per_time  # 整个序列的loss的均值

        #pred = torch.stack(pred_list).permute(1, 0)
        prob = torch.stack(prob_list).squeeze(1)

        return  prob
       #有10个list， 每个list是一个（1，5）tensor



###SHAP
class shap_normalGRUpertime(nn.Module):
    def __init__(self, normalGRUpertime_params, short):
        super(shap_normalGRUpertime, self).__init__()
        self.input_dim=normalGRUpertime_params['input_dim']
        self.n_units=normalGRUpertime_params['n_units']
        self.depth = normalGRUpertime_params['time_depth']
        self.output_dim = normalGRUpertime_params['output_dim']
        self.bias = normalGRUpertime_params['bias']
        self.num_classes = normalGRUpertime_params['num_classes']

        self.U_r = nn.Parameter(torch.randn(self.input_dim, self.n_units)  ) #正态分布
        self.W_r = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.b_r = nn.Parameter(torch.zeros(self.n_units))
        self.W_z = nn.Parameter(torch.randn(self.n_units, self.n_units)  ) #输入维度，输出维度
        self.U_z=nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_z = nn.Parameter(torch.zeros(self.n_units))
        self.W_h = nn.Parameter(torch.randn(self.n_units, self.n_units)  )  # 输入维度，输出维度
        self.U_h = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_h = nn.Parameter(torch.zeros(self.n_units))
        self.W_p=nn.Parameter(torch.randn(self.n_units, self.output_dim))
        self.b_p= nn.Parameter(torch.zeros(self.output_dim))

        self.classify = nn.Sequential(
            torch.nn.Linear(self.n_units, self.output_dim, bias=self.bias),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
            for name, param in self.named_parameters():
                if 'B' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'U' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'W' in name:
                    nn.init.orthogonal_(param.data)
                elif 'b' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'u' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'w' in name:
                    nn.init.orthogonal_(param.data)

    def forward(self, x):
        device = torch.device("cuda:1")
        h_tilda_t = torch.zeros(x.shape[0], self.n_units).to(device)
        end = x.shape[1]
        for t in range(x.shape[1]):
            r_t=torch.sigmoid(((x[:,t,:]@self.U_r)+(h_tilda_t@self.W_r)+self.b_r))
            z_t = torch.sigmoid(((x[:, t, :] @ self.U_z) + (h_tilda_t @ self.W_z) + self.b_z))
            g_t = torch.tanh(((x[:, t, :] @ self.U_h) + (h_tilda_t * r_t)@self.W_h + self.b_h))
            h_tilda_t=z_t*h_tilda_t+(1-z_t)*g_t
            if (t==end-1):
                pred_y = (self.classify(h_tilda_t))

        return pred_y



######LRP
from numpy import newaxis as na

def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=0.0, debug=False):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
    - eps:            stabilizer (small positive number)
    - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
    sign_out = np.where(hout[na, :] >= 0, 1., -1.)  # shape (1, M)

    numer = (w * hin[:, na]) + (bias_factor * (b[na, :] * 1. + eps * sign_out * 1.) / bias_nb_units)  # shape (D, M)
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)

    denom = hout[na, :] + (eps * sign_out * 1.)  # shape (1, M)

    message = (numer / denom) * Rout[na, :]  # shape (D, M)

    Rin = message.sum(axis=1)  # shape (D,)

    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())
    # Note:
    # - local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
    # - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections
    # -> can be used for sanity check

    return Rin

class lrp_normalGRUpertime(torch.jit.ScriptModule):
    def __init__(self, normalGRUpertime_params, short):
        super(lrp_normalGRUpertime, self).__init__()
        self.input_dim=normalGRUpertime_params['input_dim']
        self.n_units=normalGRUpertime_params['n_units']
        self.depth = normalGRUpertime_params['time_depth']
        self.output_dim = normalGRUpertime_params['output_dim']
        self.bias = normalGRUpertime_params['bias']
        self.num_classes = normalGRUpertime_params['num_classes']

        self.U_r = nn.Parameter(torch.randn(self.input_dim, self.n_units)  ) #正态分布
        self.W_r = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.b_r = nn.Parameter(torch.zeros(self.n_units))
        self.W_z = nn.Parameter(torch.randn(self.n_units, self.n_units)  ) #输入维度，输出维度
        self.U_z=nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_z = nn.Parameter(torch.zeros(self.n_units))
        self.W_h = nn.Parameter(torch.randn(self.n_units, self.n_units)  )  # 输入维度，输出维度
        self.U_h = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_h = nn.Parameter(torch.zeros(self.n_units))
        self.W_p=nn.Parameter(torch.randn(self.n_units, self.output_dim))
        self.b_p= nn.Parameter(torch.zeros(self.output_dim))
        self.short = short
        self.classify = nn.Sequential(
            torch.nn.Linear(self.n_units, self.output_dim, bias=self.bias),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
            for name, param in self.named_parameters():
                if 'B' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'U' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'W' in name:
                    nn.init.orthogonal_(param.data)
                elif 'b' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'u' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'w' in name:
                    nn.init.orthogonal_(param.data)

    def forward(self, x):
        device=torch.device("cuda:1")
        #h_tilda_t = torch.zeros(x.shape[0], self.n_units).to(device)
        h_tilda_t = torch.zeros(x.shape[0],self.n_units).to(device)
        end=x.shape[1]
        #self.h=torch.zeros((x.shape[0],end,self.n_units))
        #self.r=torch.zeros((x.shape[0],end-1,self.n_units))
        #self.z=torch.zeros((x.shape[0],end-1,self.n_units))
        #self.g=torch.zeros((x.shape[0],end-1,self.n_units))
        #self.preg=torch.zeros((x.shape[0],end-1,self.n_units))
        self.h = torch.zeros((end, self.n_units))
        self.r = torch.zeros(( end - 1, self.n_units))
        self.z = torch.zeros(( end - 1, self.n_units))
        self.g = torch.zeros(( end - 1, self.n_units))
        self.preg = torch.zeros((end - 1, self.n_units))

        for t in range(end):
            r_t=torch.sigmoid(((x[:,t,:]@self.U_r)+(h_tilda_t@self.W_r)+self.b_r))
            z_t = torch.sigmoid(((x[:, t, :] @ self.U_z) + (h_tilda_t @ self.W_z) + self.b_z))
            g_t = torch.tanh(((x[:, t, :] @ self.U_h) + (h_tilda_t * r_t)@self.W_h + self.b_h))
            h_tilda_t=z_t*h_tilda_t+(1-z_t)*g_t
            self.h[t]=h_tilda_t
            if t<end-1:
               # self.r[:,t,:]=r_t
               # self.z[:,t,:]=z_t
               # self.g[:,t,:]=g_t
               # self.preg[:,t,:]=((x[:, t, :] @ self.U_h) + (h_tilda_t * r_t)@self.W_h + self.b_h)
                self.r[ t, :] = r_t
                self.z[ t, :] = z_t
                self.g[ t, :] = g_t
                self.preg[t, :] = ((x[ t, :] @ self.U_h) + (h_tilda_t * r_t) @ self.W_h + self.b_h)
            if (t ==end-1):
                pred_y = (self.classify(h_tilda_t))
                self.predy=pred_y
###pred_y shape 128,5
        return pred_y

    #lrp_class 预测的类别 ,X 应该是整个序列输入，LRP-class,只是最后一个时刻的输入 ，因为最后用ht往回倒
    def lrp(self,x,LRP_class,eps=0.001, bias_factor=0.0): #LRP_class 预测的类别
        ##relevance 初始化,给定R-out,返回Rx，Rh（t-1）
        depth=x.shape[1]
        Rx=np.zeros(x.shape)
        Rh=np.zeros((x.shape[0],depth,self.n_units)) #3852,11,32
        Rg=np.zeros((x.shape[0],depth-1,self.n_units)) #表门htilda  3852,10,64
        Rout_mask=np.zeros((x.shape[0],self.output_dim))#3852,5

        Rh = np.zeros(( depth, self.n_units))  # 3852,11,32
        Rg = np.zeros((depth - 1, self.n_units))  # 表门htilda  3852,10,64
        Rout_mask = np.zeros(( self.output_dim))

        Rout_mask[LRP_class] = 1.0

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        #hin=h_t, w=和h_t对应的参数，shape C*d，是最后一层分类函数的权重，hout是输出
        w=self.classify[0].weight.data  #shape 5 64 tensor
        b=self.classify[0].bias.data #shape 5
        #最后一层分类的lrp
        aa = self.h[ depth - 2, :].detach().cpu().numpy()
        ee=self.predy
        bb = self.predy.detach().cpu().numpy()
        #cc = self.predy*Rout_mask
        dd = self.predy.detach().cpu().numpy()*Rout_mask
   ##是不是应该LRP*predy

        Rh[:,depth - 2] = lrp_linear(self.h[depth-2,:].detach().cpu().numpy(),
                                     w,b,self.predy.detach().cpu().numpy(),self.predy*Rout_mask,
                                     self.n_units,eps,bias_factor,debug=False)

        #逆回去算gru
        for t in reversed(range(depth-1)):
            Rg[t]=lrp_linear((1-self.z[t])*self.g[t],np.identity(self.n_units),np.zeros(self.n_units),
                             self.h[t],Rh[t],self.n_units+self.n_units,eps,bias_factor,debug=False)
            Rh[t-1]=lrp_linear(self.z[t]*self.h[t-1],np.identity(self.n_units),np.zeros(self.n_units),
                                self.h[t],Rh[t],self.n_units+self.n_units,eps,bias_factor,debug=False)
            Rx[t]=lrp_linear(x[t],self.U_h,self.r[t]*self.b_h,self.preg[t],Rg[t],self.input_dim+self.output_dim,eps,bias_factor,debug=False)

            Rh[t-1]+=lrp_linear(self.h[t-1],self.r[t]*self.W_h,self.r[t]*self.b_h,self.preg[t],Rg[t],self.input_dim+self.n_units,
                                eps,bias_factor,debug=False)

        return Rh, Rx

    ##最后Rx，代表是的每个时刻的变量的relevance，在每个时刻都会有后退到1


class new_two_Delegru(torch.jit.ScriptModule):
    def __init__(self, Delegru_params, short):
        super(new_two_Delegru, self).__init__()
        self.input_dim_s=Delegru_params['input_dim_s']
        self.input_dim_l = Delegru_params['input_dim_l']
        self.n_units=Delegru_params['n_units']
        self.depth = Delegru_params['time_depth']
        self.output_dim = Delegru_params['output_dim']
        self.num_classes = Delegru_params['num_classes']
        self.N_units = Delegru_params['n_units']
        self.bias = Delegru_params['bias']
        self.short=short

        #shop tensor LSTM parameter
        #torch.set_default_dtype(torch.float64)
        self.U_r_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.U_z_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.U_h_s = nn.Parameter(torch.randn(self.input_dim_s, 1, self.n_units))
        self.W_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.W_z_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.W_h_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units, self.n_units))
        self.B_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.n_units))
        self.B_z_s = nn.Parameter(torch.Tensor(self.input_dim_s, self.n_units))
        self.B_h_s = nn.Parameter(torch.Tensor(self.input_dim_s, self.n_units))

        #shop vanilla LSTM parameter
        self.u_r_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units) )  # 正态分布
        self.w_r_s = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.b_r_s = nn.Parameter(torch.zeros(self.N_units))
        self.w_z_s = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_z_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units) )
        self.b_z_s = nn.Parameter(torch.zeros(self.N_units))
        self.w_h_s = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_h_s = nn.Parameter(torch.randn(self.input_dim_s, self.N_units) )
        self.b_h_s = nn.Parameter(torch.zeros(self.N_units))
        self.w_p_s = nn.Parameter(torch.randn(self.N_units, self.output_dim))
        self.b_p_s = nn.Parameter(torch.zeros(self.output_dim))

        #loan tensor LSTM parameter
        self.U_r_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.U_z_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.U_h_l = nn.Parameter(torch.randn(self.input_dim_l, 1, self.n_units))
        self.W_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.W_z_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.W_h_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units, self.n_units))
        self.B_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.n_units))
        self.B_z_l = nn.Parameter(torch.Tensor(self.input_dim_l, self.n_units))
        self.B_h_l = nn.Parameter(torch.Tensor(self.input_dim_l, self.n_units))

        #loan vanilla LSTM parameter

        self.u_r_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )  # 正态分布
        self.w_r_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.b_r_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_z_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_z_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )
        self.b_z_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_h_l = nn.Parameter(torch.randn(self.N_units, self.N_units) )  # 输入维度，输出维度
        self.u_h_l = nn.Parameter(torch.randn(self.input_dim_l, self.N_units) )
        self.b_h_l = nn.Parameter(torch.zeros(self.N_units))
        self.w_p_l = nn.Parameter(torch.randn(self.N_units, self.output_dim))
        self.b_p_l = nn.Parameter(torch.zeros(self.output_dim))

        self.classify = torch.nn.Linear(self.N_units*2, self.output_dim, bias=self.bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B_f' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'b_f' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'B' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                #nn.init.orthogonal_(param.data)
                nn.init.xavier_uniform_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                #nn.init.orthogonal_(param.data)
                nn.init.xavier_uniform_(param.data)

    def forward(self, x,target, device):
        H_tilda_t_s = (torch.zeros(x.shape[0], self.input_dim_s, self.n_units)).to(device)
        H_tilda_t_l = (torch.zeros(x.shape[0], self.input_dim_l, self.n_units)).to(device)

        # 每个时刻对应所有feature的hidden
        h_tilda_t_s = torch.zeros(x.shape[0],  self.N_units).to(device)
        h_tilda_t_l = torch.zeros(x.shape[0], self.N_units).to(device)

        unorm_list_s = torch.jit.annotate(list[Tensor], [])
        unorm_list_l = torch.jit.annotate(list[Tensor], [])
        weights_list = torch.jit.annotate(list[Tensor], [])

        pred_list = torch.jit.annotate(list[Tensor], [])
        prob_list = torch.jit.annotate(list[Tensor], [])
        weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
        loss_CE = torch.as_tensor(0.0)
        #print('new begin')
        for t in range(self.depth-1):
            #先算shop
            # eq 5
            R_tilda_t_s = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_s, self.W_r_s) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :self.input_dim_s].unsqueeze(1), self.U_r_s) + self.B_r_s)
            Z_tilda_t_s = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_s, self.W_z_s) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :self.input_dim_s].unsqueeze(1), self.U_z_s) + self.B_z_s)
            G_tilda_t_s = torch.tanh(torch.einsum("bij,ijk->bik",(H_tilda_t_s * R_tilda_t_s), self.W_h_s) + \
                                      torch.einsum("bij,jik->bjk", x[:, t,:self.input_dim_s ].unsqueeze(1), self.U_h_s) + self.B_h_s)

            H_tilda_t_s = Z_tilda_t_s * H_tilda_t_s+(1-Z_tilda_t_s)*G_tilda_t_s #shape batch, feature, dim

            #所有lstm的关系 shop
            r_t_s = torch.sigmoid(((x[:, t, :self.input_dim_s] @ self.u_r_s) + (h_tilda_t_s @ self.w_r_s) + self.b_r_s))
            z_t_s = torch.sigmoid(((x[:, t, :self.input_dim_s] @ self.u_z_s) + (h_tilda_t_s @ self.w_z_s) + self.b_z_s))
            g_t_s = torch.tanh(((x[:, t, :self.input_dim_s] @ self.u_h_s) + ((h_tilda_t_s *r_t_s) @ self.w_h_s) + self.b_h_s))
            h_tilda_t_s = z_t_s * h_tilda_t_s +(1-z_t_s)*g_t_s

            #tensor lstm loan
            R_tilda_t_l = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_l, self.W_r_l) + \
                                        torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                     self.U_r_l) + self.B_r_l)
            Z_tilda_t_l = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t_l, self.W_z_l) + \
                                        torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                     self.U_z_l) + self.B_z_l)
            G_tilda_t_l = torch.tanh(torch.einsum("bij,ijk->bik", (H_tilda_t_l * R_tilda_t_l), self.W_h_l) + \
                                     torch.einsum("bij,jik->bjk", x[:, t, self.input_dim_s:].unsqueeze(1),
                                                  self.U_h_l) + self.B_h_l)

            H_tilda_t_l = Z_tilda_t_l * H_tilda_t_l + (1 - Z_tilda_t_l) * G_tilda_t_l  # shape batch, feature, dim

            # 所有lstm的关系 shop
            r_t_l = torch.sigmoid(((x[:, t, self.input_dim_s:] @ self.u_r_l) + (h_tilda_t_l @ self.w_r_l) + self.b_r_l))
            z_t_l = torch.sigmoid(((x[:, t, self.input_dim_s:] @ self.u_z_l) + (h_tilda_t_l @ self.w_z_l) + self.b_z_l))
            g_t_l = torch.tanh(
                ((x[:, t, self.input_dim_s:] @ self.u_h_l) + ((h_tilda_t_l * r_t_l) @ self.w_h_l) + self.b_h_l))

            h_tilda_t_l = z_t_l * h_tilda_t_l + (1 - z_t_l) * g_t_l

            if(t>self.short):

                weight_s= _estimate_alpha1(H_tilda_t_s, targets=h_tilda_t_s) #shape batch, feature 1,

                weight_l = _estimate_alpha1(H_tilda_t_l, targets=h_tilda_t_l)  # shape batch, feature 1,

                #近似值
                h_tilda_t_s=torch.bmm(H_tilda_t_s.permute(0,2,1),weight_s).squeeze(-1)

                h_tilda_t_l = torch.bmm(H_tilda_t_l.permute(0, 2, 1), weight_l).squeeze(-1)

                unorm_weight_s=weight_s
                unorm_weight_l = weight_l

                newhtilda_t=torch.concat([h_tilda_t_s,h_tilda_t_l],dim=1)

                pred_y = (self.classify(newhtilda_t)) #shape  batch, 2

                #shop weights, loan weights
                all_weights=self.classify.weight.permute(1,0)
                all_weights_s=torch.norm(all_weights[:self.n_units,:]).unsqueeze(0)
                all_weights_l=torch.norm(all_weights[self.n_units:,:]).unsqueeze(0)
                all_weights=torch.concat([all_weights_s,all_weights_l])
                # 保存每次的预测值
                loss_CE_per_time = torch.nn.CrossEntropyLoss(weight=weights)(pred_y, target[:, t - 1].long())
                loss_CE = loss_CE + loss_CE_per_time  # 整个序列的loss的均值

                # 保存每次的预测值
                softmax = torch.nn.Softmax(dim=1)
                pred_y = softmax(pred_y)
                prediction = torch.argmax(pred_y, dim=1)  # shape batch
                prob_list += [pred_y[:, 1]]  # 这里为了计算auc，保存类别为1的概率值
                pred_list += [prediction]

                unorm_list_s += [unorm_weight_s]
                unorm_list_l += [unorm_weight_l]
                weights_list+=[all_weights]

        pred = torch.stack(pred_list).permute(1,0)
        unorm_s = torch.stack(unorm_list_s) #shape time_depth, BATCH input_dim
        unorm_l = torch.stack(unorm_list_l)
        weights=torch.stack(weights_list)

        return loss_CE, pred,None, unorm_s, unorm_l,weights,None








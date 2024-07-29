import os
import time
import numpy as np
import pandas as pd
import torch
from torch import  Tensor
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
import warnings
import shap
import quantus
import torch.nn as nn
warnings.filterwarnings('ignore')

class ModelWrapper(nn.Module):
    def __init__(self, model, target, device):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.target = target
        self.device = device

    def forward(self, x):
        return self.model(x, target=self.target, device=self.device)


def gridsearch_training(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id,units,lr,batch_size):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=20, verbose=True,
                                                                 min_lr=0.0000001)
    epochs =40
    max_acc = 0
    save_path = os.path.join('original_grid_search', f'{units}{lr}{batch_size}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        j = 0
        true_train = []
        preds_train = []
        for batch_x, batch_y in train_loader:
            j += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape batch, depth
            opt.zero_grad()
            loss_CE, output, _, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            true_train.append(batch_y.detach().cpu().numpy())
            preds_train.append(output.detach().cpu().numpy())
            l = loss_CE
            l.backward()
            opt.step()
            train_loss += l.item()

        true_train = np.concatenate(true_train).reshape(-1)
        preds_train = np.concatenate(preds_train).reshape(-1)

        # 计算各种指标
        acc_train = accuracy_score(true_train, preds_train)
        pre_train = precision_score(true_train, preds_train, average='macro')
        recall_train = recall_score(true_train, preds_train, average='macro')
        f1_train = f1_score(true_train, preds_train, average='macro')
        kp_train = cohen_kappa_score(true_train, preds_train)

        model_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        time_elapsed = time.time() - epoch_start_time

        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train loss'] = train_loss / (j)
            df_log_val.loc[i, 'Train Acc'] = acc_train
            df_log_val.loc[i, 'Train Precision'] = pre_train
            df_log_val.loc[i, 'Train Recall'] = recall_train
            df_log_val.loc[i, 'Train F-measure'] = f1_train
            df_log_val.loc[i, 'Train KP'] = kp_train

        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())

        preds_val = np.concatenate(preds).reshape(-1)
        true_val = np.concatenate(true).reshape(-1)

        val_result = classification_report(true_val, preds_val, output_dict=True)
        val_result = pd.DataFrame(val_result).transpose()

        acc_val = accuracy_score(true_val, preds_val)
        pre_val = precision_score(true_val, preds_val, average='macro')
        recall_val = recall_score(true_val, preds_val, average='macro')
        f1_val = f1_score(true_val, preds_val, average='macro')
        kp_val = cohen_kappa_score(true_val, preds_val)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if i == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))
        if acc_val > max_acc:
            max_acc = acc_val
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))
        if args.log:
            df_log_val.loc[i, 'Val Acc'] = acc_val
            df_log_val.loc[i, 'Val Precision'] = pre_val
            df_log_val.loc[i, 'Val Recall'] = recall_val
            df_log_val.loc[i, 'Val F-measure'] = f1_val
            df_log_val.loc[i, 'Val KP'] = kp_val

        epoch_scheduler.step(acc_val)

    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt')))
        preds = []
        true = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())

    preds_test = np.concatenate(preds).reshape(-1)
    true_test = np.concatenate(true).reshape(-1)
    test_result = classification_report(true_test, preds_test, output_dict=True)
    test_result = pd.DataFrame(test_result).transpose()

    acc_test = accuracy_score(true_test, preds_test)
    pre_test = precision_score(true_test, preds_test, average='macro')
    recall_test = recall_score(true_test, preds_test, average='macro')
    f1_test = f1_score(true_test, preds_test, average='macro')
    kp_test = cohen_kappa_score(true_test, preds_test)

    df_log_test.loc[0, 'acc_test'] = acc_test
    df_log_test.loc[0, 'precision_test'] = pre_test
    df_log_test.loc[0, 'recall_test'] = recall_test
    df_log_test.loc[0, 'f1_test'] = f1_test
    df_log_test.loc[0, 'kp_test'] = kp_test
    print('acc_test, prec_test, recall_test,f1_test,kp_test,hidden,lr,batch_size',acc_test, pre_test, recall_test,f1_test,
          kp_test,units,lr,batch_size)

    outcome = np.vstack((true_test, preds_test))
    outcome = pd.DataFrame(outcome)
    time_final = time.time() - epoch_start_time
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path, str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    val_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'val_report_results.csv'))
    test_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'test_report_results.csv'))
    print('模型训练时间：{}'.format(time_elapsed))
    print('模型运行时间：{}'.format(time_final))

def credit_time(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=20, verbose=True,
                                                                 min_lr=0.0000001)
    epochs = 40
    max_acc = 0

    save_path = args.save_dirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        j = 0
        true_train = []
        preds_train = []
        for batch_x, batch_y in train_loader:
            j += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape batch, depth
            opt.zero_grad()

            loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            true_train.append(batch_y.detach().cpu().numpy())
            preds_train.append(output.detach().cpu().numpy())

            l = loss_CE
            l.backward()
            opt.step()
            train_loss += l.item()
            #print(l.item())

        true_train = np.concatenate(true_train).reshape(-1)
        preds_train = np.concatenate(preds_train).reshape(-1)

        #计算指标
        acc_train = accuracy_score(true_train, preds_train)
        pre_train = precision_score(true_train, preds_train, average='macro')
        recall_train = recall_score(true_train, preds_train, average='macro')
        f1_train = f1_score(true_train, preds_train, average='macro')
        kp_train = cohen_kappa_score(true_train, preds_train)

        model_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        time_elapsed = time.time() - epoch_start_time
        #print(f"Training loss at epoch {i}: loss={train_loss / j:.4f} "
         #f'kp_train :{kp_train: .2f}'
          #   f"time elapsed: {time_elapsed:.2f}")

        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train loss'] = train_loss / (j)
            df_log_val.loc[i, 'Train Acc'] = acc_train
            df_log_val.loc[i, 'Train Precision'] = pre_train
            df_log_val.loc[i, 'Train Recall'] = recall_train
            df_log_val.loc[i, 'Train F-measure'] = f1_train
            df_log_val.loc[i, 'Train KP'] = kp_train

        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            val_alpha = []
            val_beta = []
            valweight_s_list = torch.jit.annotate(list[Tensor], [])
            valweight_l_list = torch.jit.annotate(list[Tensor], [])
            retain_valweight_s_list = torch.jit.annotate(list[Tensor], [])

            weights_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())

                if model_name in ['IMV_full', 'IMV_tensor']:
                    val_alpha.append(unorm_s.detach().cpu().numpy())
                    val_beta.append(unorm_l.detach().cpu().numpy())
                if model_name in ['GLEN']:
                    unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                    valweight_s_list += [unorm_s]
                    unorm_l = unorm_l.squeeze(-1)  # shape time depth, batch, feature
                    valweight_l_list += [unorm_l]
                    weights_list += [weights]

                if model_name in ['Retain']:
                    retain_valweight_s_list += [unorm_s]
        if model_name in ['GLEN']:
            valweight_s = torch.stack(valweight_s_list)
            valweight_s = valweight_s.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_s )
            valweight_s = valweight_s.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight_s = pd.DataFrame(valweight_s)

            valweight_l = torch.stack(valweight_l_list)
            valweight_l = valweight_l.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_l )
            valweight_l = valweight_l.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight_l = pd.DataFrame(valweight_l)

            valweights = torch.stack(weights_list)
            valweights = valweights.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweights = pd.DataFrame(valweights)

        if model_name in ['IMV_full', 'IMV_tensor']:
            val_alpha = np.concatenate(val_alpha)
            val_beta = np.concatenate(val_beta)
            val_alpha = val_alpha.mean(axis=0)
            val_beta = val_beta.mean(axis=0)
            val_beta = pd.DataFrame(val_beta)
            val_alpha = pd.DataFrame(val_alpha)
        if model_name in ['Retain']:
            retain_valweight = torch.stack(retain_valweight_s_list)
            retain_valweight = retain_valweight.reshape(-1, args.depth - 1, args.input_dim_s + args.input_dim_l)
            retain_valweight = retain_valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            retain_valweight = pd.DataFrame(retain_valweight)
        preds_val = np.concatenate(preds).reshape(-1)
        true_val = np.concatenate(true).reshape(-1)
        val_result = classification_report(true_val, preds_val, output_dict=True)
        val_result = pd.DataFrame(val_result).transpose()

        acc_val = accuracy_score(true_val, preds_val)
        pre_val = precision_score(true_val, preds_val, average='macro')
        recall_val = recall_score(true_val, preds_val, average='macro')
        f1_val = f1_score(true_val, preds_val, average='macro')
        kp_val = cohen_kappa_score(true_val, preds_val)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if i==0:
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))
        if acc_val > max_acc:
            max_acc = acc_val
            print("Saving...")

            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))
            if model_name in ['IMV_full', 'IMV_tensor']:
                val_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_valalpha.csv'))
                val_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_valbeta.csv'))
            if model_name in ['GLEN']:
                valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_shop.csv'))
                valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_loan.csv'))
                valweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_twoparts.csv'))
                soft_valweight_s = valweight_s.copy()

                for row in range(len(soft_valweight_s)):
                    soft_valweight_s.loc[row, :] = soft_valweight_s.loc[row, :].abs() / sum(
                        soft_valweight_s.loc[row, :].abs())

                soft_valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_valweight_shop.csv'))
                soft_valweight_l = valweight_l.copy()

                for row in range(len(soft_valweight_l)):
                    soft_valweight_l.loc[row, :] = soft_valweight_l.loc[row, :].abs() / sum(
                        soft_valweight_l.loc[row, :].abs())

                soft_valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_valweight_loan.csv'))

            if model_name == 'Retain':
                retain_valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))

    #    if (i % 10 == 0):
      #      print("lr: ", opt.param_groups[0]["lr"])
    #    print("Iter: ", i, "train: ", train_loss / (j), 'train_recall: ', recall_train, 'train_kp: ', kp_train,
       #       "val_recall: ", recall_val, 'val_kp', kp_val)

        if args.log:
            df_log_val.loc[i, 'Val Acc'] = acc_val
            df_log_val.loc[i, 'Val Precision'] = pre_val
            df_log_val.loc[i, 'Val Recall'] = recall_val
            df_log_val.loc[i, 'Val F-measure'] = f1_val
            df_log_val.loc[i, 'Val kp'] = kp_val

        epoch_scheduler.step(acc_val)

    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt')))
        preds = []
        true = []
        test_alpha = []
        test_beta = []

        testweight_s_list = torch.jit.annotate(list[Tensor], [])
        testweight_l_list = torch.jit.annotate(list[Tensor], [])

        retain_testweight_s_list = torch.jit.annotate(list[Tensor], [])

        weights_list = torch.jit.annotate(list[Tensor], [])
        # unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            # loss_test, unorm_weight,output,prob,_= model(batch_x.float(), batch_y,device)  # unorm time, batch, features
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())

            if model_name in ['IMV_full', 'IMV_tensor']:
                test_alpha.append(unorm_s.detach().cpu().numpy())
                test_beta.append(unorm_l.detach().cpu().numpy())
            if model_name in ['GLEN']:
                unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                testweight_s_list += [unorm_s]
                unorm_l = unorm_l.squeeze(-1)  # shape time depth, batch, feature
                testweight_l_list += [unorm_l]
                weights_list += [weights]

            if model_name in ['Retain']:
                # shape time depth, batch, feature
                retain_testweight_s_list += [unorm_s]

    preds_test = np.concatenate(preds).reshape(-1)
    true_test = np.concatenate(true).reshape(-1)
    test_result = classification_report(true_test, preds_test, output_dict=True)
    test_result = pd.DataFrame(test_result).transpose()

    acc_test = accuracy_score(true_test, preds_test)
    pre_test = precision_score(true_test, preds_test, average='macro')
    recall_test = recall_score(true_test, preds_test, average='macro')
    f1_test = f1_score(true_test, preds_test, average='macro')
    kp_test = cohen_kappa_score(true_test, preds_test)

    if model_name in ['Retain']:
        retain_test_weight = torch.stack(retain_testweight_s_list)  # shape 19, 5, 100, 5
        retain_test_weight = retain_test_weight.reshape(-1, args.depth - 1, args.input_dim_s + args.input_dim_l)
        retain_test_weight = retain_test_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        retain_test_weight = pd.DataFrame(retain_test_weight)
        retain_test_weight.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight.csv'))
    if model_name in ['GLEN']:
        testweight_s = torch.stack(testweight_s_list)
        testweight_s = testweight_s.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_s )
        testweight_s = testweight_s.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweight_s = pd.DataFrame(testweight_s)

        testweight_l = torch.stack(testweight_l_list)
        testweight_l = testweight_l.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_l)
        testweight_l = testweight_l.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweight_l = pd.DataFrame(testweight_l)

        testweights = torch.stack(weights_list)
        testweights = testweights.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweights = pd.DataFrame(testweights)

        testweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_shop.csv'))
        testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_loan.csv'))
        testweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_twoparts.csv'))
        soft_testweight_s = testweight_s.copy()
        for i in range(len(soft_testweight_s)):
            soft_testweight_s.loc[i, :] = soft_testweight_s.loc[i, :].abs() / sum(soft_testweight_s.loc[i, :].abs())

        soft_testweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight_shop.csv'))

        soft_testweight_l = testweight_l.copy()
        for i in range(len(soft_testweight_l)):
            soft_testweight_l.loc[i, :] = soft_testweight_l.loc[i, :].abs() / sum(soft_testweight_l.loc[i, :].abs())

        soft_testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight_loan.csv'))

    if model_name in ['IMV_full', 'IMV_tensor']:
        test_alpha = np.concatenate(test_alpha)
        test_beta = np.concatenate(test_beta)
        test_alpha = test_alpha.mean(axis=0)
        test_beta = test_beta.mean(axis=0)
        test_beta = pd.DataFrame(test_beta)
        test_alpha = pd.DataFrame(test_alpha)
        test_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_testalpha.csv'))
        test_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_testbeta.csv'))

    df_log_test.loc[0, 'acc_test'] = acc_test
    df_log_test.loc[0, 'precision_test'] = pre_test
    df_log_test.loc[0, 'recall_test'] = recall_test
    df_log_test.loc[0, 'f1_test'] = f1_test
    df_log_test.loc[0, 'k_test'] = kp_test

    print('acc_test, prec_test, recall_test,f1_test,kp_test',acc_test, pre_test, recall_test,f1_test,
          kp_test)

    outcome = np.vstack((true_test, preds_test))
    outcome = pd.DataFrame(outcome)
    time_final=time.time()-epoch_start_time
    #
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path, str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    val_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'val_report_results.csv'))
    test_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'test_report_results.csv'))
    print('模型训练时间：{}'.format(time_elapsed))
    print('模型训运行间：{}'.format(time_final))

###解释model
def shap_lrp_new_credit_training2(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id,X_train_t,X_test_t):
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=20, verbose=True,
                                                                 min_lr=0.0000001)
    epochs = 40
    max_acc = 0

    save_path = args.save_dirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    start_time=time.time()
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        j = 0
        true_train = []
        prob_list_train = []
        preds_train = []
        for batch_x, batch_y in train_loader:
            j += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape batch, depth
            opt.zero_grad()
            loss_per=0
            weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
            pred_list = torch.jit.annotate(list[Tensor], [])
            criterion=torch.nn.CrossEntropyLoss(weight=weights)
            for number in range(1,8):
                prob = model(batch_x[:,:number].float())
                loss_per=loss_per+criterion(prob,batch_y[:,number-1].long())
                prediction=torch.argmax(prob,dim=1)
                pred_list+=[prediction]
            pred = torch.stack(pred_list).permute(1, 0)
            true_train.append(batch_y.detach().cpu().numpy())
            preds_train.append(pred.detach().cpu().numpy())

            l = loss_per
            l.backward()
            opt.step()
            train_loss += l.item()

        true_train = np.concatenate(true_train).reshape(-1)
        preds_train = np.concatenate(preds_train).reshape(-1)

        #计算指标
        acc_train = accuracy_score(true_train, preds_train)
        pre_train = precision_score(true_train, preds_train, average='macro')
        recall_train = recall_score(true_train, preds_train, average='macro')
        f1_train = f1_score(true_train, preds_train, average='macro')
        kp_train = cohen_kappa_score(true_train, preds_train)
        model_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train loss'] = train_loss / (j)
            df_log_val.loc[i, 'Train Acc'] = acc_train
            df_log_val.loc[i, 'Train Precision'] = pre_train
            df_log_val.loc[i, 'Train Recall'] = recall_train
            df_log_val.loc[i, 'Train F-measure'] = f1_train
            df_log_val.loc[i, 'Train KP'] = kp_train

        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                prob = model(batch_x.float())
                loss_per = 0
                weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
                pred_list = torch.jit.annotate(list[Tensor], [])
                #pred_list=[]
                criterion = torch.nn.CrossEntropyLoss(weight=weights)
                for number in range(1, 8):
                    prob = model(batch_x[:, :number].float())
                    # prob_time=prob[:,number,:] #shape batch, 5
                    loss_per = loss_per + criterion(prob, batch_y[:, number - 1].long())
                    prediction = torch.argmax(prob, dim=1)
                    pred_list += [prediction]
                pred= torch.stack(pred_list).permute(1, 0)
                true.append(batch_y.detach().cpu().numpy())
                preds.append(pred.detach().cpu().numpy())
        preds_val = np.concatenate(preds).reshape(-1)
        true_val = np.concatenate(true).reshape(-1)

        val_result = classification_report(true_val, preds_val, output_dict=True)
        val_result = pd.DataFrame(val_result).transpose()

        acc_val = accuracy_score(true_val, preds_val)
        pre_val = precision_score(true_val, preds_val, average='macro')
        recall_val = recall_score(true_val, preds_val, average='macro')
        f1_val = f1_score(true_val, preds_val, average='macro')
        kp_val = cohen_kappa_score(true_val, preds_val)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if i==0:
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

        if acc_val > max_acc:
            max_acc = acc_val
            print("Saving...")
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

        if args.log:
            df_log_val.loc[i, 'Val Acc'] = acc_val
            df_log_val.loc[i, 'Val Precision'] = pre_val
            df_log_val.loc[i, 'Val Recall'] = recall_val
            df_log_val.loc[i, 'Val F-measure'] = f1_val
            df_log_val.loc[i, 'Val kp'] = kp_val

        epoch_scheduler.step(acc_val)

    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt')))

        preds = []
        true = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prob= model(batch_x.float())
            loss_per = 0
            weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
            pred_list = torch.jit.annotate(list[Tensor], [])
            #pred_list=[]
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
            for number in range(1,8):
                prob = model(batch_x[:,:number].float())
                #prob_time=prob[:,number,:] #shape batch, 5
                loss_per=loss_per+criterion(prob,batch_y[:,number-1].long())
                prediction=torch.argmax(prob,dim=1)
                pred_list+=[prediction]
            pred = torch.stack(pred_list).permute(1, 0)
            true.append(batch_y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

    preds_test = np.concatenate(preds).reshape(-1)
    true_test = np.concatenate(true).reshape(-1)
    test_result = classification_report(true_test, preds_test, output_dict=True)
    test_result = pd.DataFrame(test_result).transpose()

    acc_test = accuracy_score(true_test, preds_test)
    pre_test = precision_score(true_test, preds_test, average='macro')
    recall_test = recall_score(true_test, preds_test, average='macro')
    f1_test = f1_score(true_test, preds_test, average='macro')
    kp_test = cohen_kappa_score(true_test, preds_test)

    df_log_test.loc[0, 'acc_test'] = acc_test
    df_log_test.loc[0, 'precision_test'] = pre_test
    df_log_test.loc[0, 'recall_test'] = recall_test
    df_log_test.loc[0, 'f1_test'] = f1_test
    df_log_test.loc[0, 'k_test'] = kp_test

    print('acc_test, prec_test, recall_test,f1_test,kp_test',acc_test, pre_test, recall_test,f1_test,
          kp_test)

    outcome = np.vstack((true_test, preds_test))
    outcome = pd.DataFrame(outcome)
    if model_name=='SHAP':
        class0_shap_values_list=[]
        class1_shap_values_list = []
        class2_shap_values_list = []
        #class3_shap_values_list = []
        #class4_shap_values_list = []
        shap_values_list = []
        for number in range(1, 8):
            explainer = shap.DeepExplainer(model.to(device), X_train_t[:,:number].float().to(device))
            shap_values = explainer.shap_values(X_test_t[:,:number].float().to(device))
            class0_shap_values_list.append(np.mean(np.mean(shap_values[0],axis=1),axis=0)) #shape 是feature
            class1_shap_values_list.append(np.mean(np.mean(shap_values[1],axis=1),axis=0))
            class2_shap_values_list.append(np.mean(np.mean(shap_values[2],axis=1),axis=0))
            #class3_shap_values_list.append(np.mean(np.mean(shap_values[3],axis=1),axis=0))
            #class4_shap_values_list.append(np.mean(np.mean(shap_values[4],axis=1),axis=0))
            shap_values_list.append(shap_values)

        class_0 = pd.DataFrame(np.stack(class0_shap_values_list,axis=0))
        class_1 =pd.DataFrame(np.stack(class1_shap_values_list,axis=0))
        class_2 = pd.DataFrame(np.stack(class2_shap_values_list,axis=0))
        #class_3 = pd.DataFrame(np.stack(class3_shap_values_list,axis=0))
        #class_4 = pd.DataFrame(np.stack(class4_shap_values_list,axis=0))
        class_0.to_csv(os.path.join(model_save_path, str(exp_id) + 'class_0.csv'))
        class_1.to_csv(os.path.join(model_save_path, str(exp_id) + 'class_1.csv'))
        class_2.to_csv(os.path.join(model_save_path, str(exp_id) + 'class_2.csv'))
        #class_3.to_csv(os.path.join(model_save_path, str(exp_id) + 'class_3.csv'))
       # class_4.to_csv(os.path.join(model_save_path, str(exp_id) + 'class_4.csv'))
    time_final=time.time()
    print('运行时间：{}'.format(time_final-start_time))


    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path, str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    val_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'val_report_results.csv'))
    test_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'test_report_results.csv'))

###不分解
def credit_training2_nodecompose_2(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=20, verbose=True,
                                                                 min_lr=0.0000001)
    epochs =40
    max_acc = 0

    save_path = os.path.join(args.save_dirs,  'exp_' + str(exp_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        j = 0
        true_train = []
        preds_train = []
        for batch_x, batch_y in train_loader:
            j += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape batch, depth
            opt.zero_grad()

            loss_CE, output, _, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

            true_train.append(batch_y.detach().cpu().numpy())
            preds_train.append(output.detach().cpu().numpy())

            l = loss_CE
            l.backward()
            opt.step()
            train_loss += l.item()

        true_train = np.concatenate(true_train).reshape(-1)
        preds_train = np.concatenate(preds_train).reshape(-1)

        # 计算各种指标
        acc_train = accuracy_score(true_train, preds_train)
        pre_train = precision_score(true_train, preds_train, average='macro')
        recall_train = recall_score(true_train, preds_train, average='macro')
        f1_train = f1_score(true_train, preds_train, average='macro')
        kp_train = cohen_kappa_score(true_train, preds_train)

        # plot the roc curve for the model
        model_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        time_elapsed = time.time() - epoch_start_time
        #print(f"Training loss at epoch {i}: loss={train_loss / j:.4f} "
         #     f'kp_train :{kp_train: .2f}'
          #    f"time elapsed: {time_elapsed:.2f}")

        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train loss'] = train_loss / (j)
            df_log_val.loc[i, 'Train Acc'] = acc_train
            df_log_val.loc[i, 'Train Precision'] = pre_train
            df_log_val.loc[i, 'Train Recall'] = recall_train
            df_log_val.loc[i, 'Train F-measure'] = f1_train
            df_log_val.loc[i, 'Train KP'] = kp_train

        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            val_alpha = []
            val_beta = []
            valweight_s_list = torch.jit.annotate(list[Tensor], [])
            valweight_l_list = torch.jit.annotate(list[Tensor], [])

            weights_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())

                if model_name in ['IMV_full', 'IMV_tensor']:
                    val_alpha.append(unorm_s.detach().cpu().numpy())
                    val_beta.append(_.detach().cpu().numpy())
                if model_name in ['GLEN']:
                    unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                    valweight_s_list += [unorm_s]
                    unorm_l = unorm_l.squeeze(-1)  # shape time depth, batch, feature
                    valweight_l_list += [unorm_l]
                    weights_list += [weights]

                if model_name in ['Retain']:
                    valweight_s_list += [unorm_s]
        if model_name in ['GLEN']:
            valweight_s = torch.stack(valweight_s_list)
            valweight_s = valweight_s.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_s )
            valweight_s = valweight_s.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight_s = pd.DataFrame(valweight_s)

            valweight_l = torch.stack(valweight_l_list)
            valweight_l = valweight_l.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_l )
            valweight_l = valweight_l.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight_l = pd.DataFrame(valweight_l)

            valweights = torch.stack(weights_list)
            valweights = valweights.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweights = pd.DataFrame(valweights)

        if model_name in ['IMV_full', 'IMV_tensor']:
            val_alpha = np.concatenate(val_alpha)
            val_beta = np.concatenate(val_beta)
            val_alpha = val_alpha.mean(axis=0)
            val_beta = val_beta.mean(axis=0)
            val_beta = pd.DataFrame(val_beta)
            val_alpha = pd.DataFrame(val_alpha)
        if model_name in ['Retain']:
            valweight = torch.stack(valweight_s_list)
            valweight = valweight.reshape(-1, args.depth - 1, args.input_dim_s+args.input_dim_l)
            valweight = valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight = pd.DataFrame(valweight)
        preds_val = np.concatenate(preds).reshape(-1)
        true_val = np.concatenate(true).reshape(-1)

        val_result = classification_report(true_val, preds_val, output_dict=True)
        val_result = pd.DataFrame(val_result).transpose()

        acc_val = accuracy_score(true_val, preds_val)
        pre_val = precision_score(true_val, preds_val, average='macro')
        recall_val = recall_score(true_val, preds_val, average='macro')
        f1_val = f1_score(true_val, preds_val, average='macro')
        kp_val = cohen_kappa_score(true_val, preds_val)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if i == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

        if acc_val > max_acc:
        #if recall_val > max_recall:
            max_acc = acc_val
            #max_recall = recall_val
            print("Saving...")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

            if model_name in ['IMV_full', 'IMV_tensor']:
                val_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_valalpha.csv'))
                val_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_valbeta.csv'))
            if model_name in['GLEN']:
                valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_shop.csv'))
                valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_loan.csv'))
                valweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_twoparts.csv'))
                soft_valweight_s = valweight_s.copy()

                for row in range(len(soft_valweight_s)):
                    soft_valweight_s.loc[row, :] = soft_valweight_s.loc[row, :].abs() / sum(
                        soft_valweight_s.loc[row, :].abs())

                soft_valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_valweight_shop.csv'))

                soft_valweight_l = valweight_l.copy()

                for row in range(len(soft_valweight_l)):
                    soft_valweight_l.loc[row, :] = soft_valweight_l.loc[row, :].abs() / sum(
                        soft_valweight_l.loc[row, :].abs())

                soft_valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_valweight_loan.csv'))

            if model_name == 'Retain':
                valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))

       # if (i % 10 == 0):
        #    print("lr: ", opt.param_groups[0]["lr"])
        #print("Iter: ", i, "train: ", train_loss / (j), 'train_recall: ', recall_train, 'train_kp: ', kp_train,
         #     "val_recall: ", recall_val, 'val_kp', kp_val)

        if args.log:
            df_log_val.loc[i, 'Val Acc'] = acc_val
            df_log_val.loc[i, 'Val Precision'] = pre_val
            df_log_val.loc[i, 'Val Recall'] = recall_val
            df_log_val.loc[i, 'Val F-measure'] = f1_val
            df_log_val.loc[i, 'Val KP'] = kp_val

        epoch_scheduler.step(acc_val)

    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt')))
        preds = []
        true = []
        test_alpha = []
        test_beta = []

        testweight_s_list = torch.jit.annotate(list[Tensor], [])
        testweight_l_list = torch.jit.annotate(list[Tensor], [])

        weights_list = torch.jit.annotate(list[Tensor], [])
        # unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            # loss_test, unorm_weight,output,prob,_= model(batch_x.float(), batch_y,device)  # unorm time, batch, features
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            if model_name in ['IMV_full', 'IMV_tensor']:
                test_alpha.append(unorm_s.detach().cpu().numpy())
                test_beta.append(_.detach().cpu().numpy())
            if model_name in ['GLEN']:
                unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                testweight_s_list += [unorm_s]
                unorm_l = unorm_l.squeeze(-1)  # shape time depth, batch, feature
                testweight_l_list += [unorm_l]
                weights_list += [weights]

            if model_name in ['Retain']:
                # shape time depth, batch, feature
                testweight_s_list += [unorm_s]

    preds_test = np.concatenate(preds).reshape(-1)
    true_test = np.concatenate(true).reshape(-1)
    test_result = classification_report(true_test, preds_test, output_dict=True)
    test_result = pd.DataFrame(test_result).transpose()

    acc_test = accuracy_score(true_test, preds_test)
    pre_test = precision_score(true_test, preds_test, average='macro')
    recall_test = recall_score(true_test, preds_test, average='macro')
    f1_test = f1_score(true_test, preds_test, average='macro')
    kp_test = cohen_kappa_score(true_test, preds_test)

    if model_name in ['Retain']:
        test_weight = torch.stack(testweight_s_list)  # shape 19, 5, 100, 5
        test_weight = test_weight.reshape(-1, args.depth - 1, args.input_dim_s+args.input_dim_l)
        test_weight = test_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        test_weight = pd.DataFrame(test_weight)
        test_weight.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight.csv'))
    if model_name in ['GLEN']:
        testweight_s = torch.stack(testweight_s_list)
        testweight_s = testweight_s.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_s )
        testweight_s = testweight_s.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweight_s = pd.DataFrame(testweight_s)

        testweight_l = torch.stack(testweight_l_list)
        testweight_l = testweight_l.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_l )
        testweight_l = testweight_l.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweight_l = pd.DataFrame(testweight_l)

        testweights = torch.stack(weights_list)
        testweights = testweights.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweights = pd.DataFrame(testweights)

        testweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_shop.csv'))
        testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_loan.csv'))
        testweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_twoparts.csv'))
        soft_testweight_s = testweight_s.copy()
        for i in range(len(soft_testweight_s)):
            soft_testweight_s.loc[i, :] = soft_testweight_s.loc[i, :].abs() / sum(soft_testweight_s.loc[i, :].abs())

        soft_testweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight_shop.csv'))

        soft_testweight_l = testweight_l.copy()
        for i in range(len(soft_testweight_l)):
            soft_testweight_l.loc[i, :] = soft_testweight_l.loc[i, :].abs() / sum(soft_testweight_l.loc[i, :].abs())

        soft_testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight_loan.csv'))

    if model_name in ['IMV_full', 'IMV_tensor']:
        test_alpha = np.concatenate(test_alpha)
        test_beta = np.concatenate(test_beta)
        test_alpha = test_alpha.mean(axis=0)
        test_beta = test_beta.mean(axis=0)
        test_beta = pd.DataFrame(test_beta)
        test_alpha = pd.DataFrame(test_alpha)
        test_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_testalpha.csv'))
        test_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_testbeta.csv'))

    df_log_test.loc[0, 'acc_test'] = acc_test
    df_log_test.loc[0, 'precision_test'] = pre_test
    df_log_test.loc[0, 'recall_test'] = recall_test
    df_log_test.loc[0, 'f1_test'] = f1_test
    df_log_test.loc[0, 'kp_test'] = kp_test


    print('acc_test, prec_test, recall_test,f1_test,kp_test',acc_test, pre_test, recall_test,f1_test,
          kp_test)


    outcome = np.vstack((true_test, preds_test))
    outcome = pd.DataFrame(outcome)
    #
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path, str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    val_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'val_report_results.csv'))
    test_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'test_report_results.csv'))

def credit_time_loan(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=20, verbose=True,
                                                                 min_lr=0.0000001)
    epochs =40
    max_acc = 0

    save_path = args.save_dirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        j = 0
        true_train = []
        preds_train = []
        for batch_x, batch_y in train_loader:
            j += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape batch, depth
            opt.zero_grad()

            loss_CE, output, _, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

            true_train.append(batch_y.detach().cpu().numpy())
            preds_train.append(output.detach().cpu().numpy())

            l = loss_CE
            l.backward()
            opt.step()
            train_loss += l.item()

        true_train = np.concatenate(true_train).reshape(-1)
        preds_train = np.concatenate(preds_train).reshape(-1)

        # 计算各种指标
        acc_train = accuracy_score(true_train, preds_train)
        pre_train = precision_score(true_train, preds_train, average='macro')
        recall_train = recall_score(true_train, preds_train, average='macro')
        f1_train = f1_score(true_train, preds_train, average='macro')
        kp_train = cohen_kappa_score(true_train, preds_train)

        # plot the roc curve for the model
        model_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        time_elapsed = time.time() - epoch_start_time
      #  print(f"Training loss at epoch {i}: loss={train_loss / j:.4f} "
      #        f'kp_train :{kp_train: .2f}'
       #       f"time elapsed: {time_elapsed:.2f}")

        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train loss'] = train_loss / (j)
            df_log_val.loc[i, 'Train Acc'] = acc_train
            df_log_val.loc[i, 'Train Precision'] = pre_train
            df_log_val.loc[i, 'Train Recall'] = recall_train
            df_log_val.loc[i, 'Train F-measure'] = f1_train
            df_log_val.loc[i, 'Train KP'] = kp_train

        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            val_alpha = []
            val_beta = []
            #valweight_s_list = torch.jit.annotate(list[Tensor], [])
            valweight_l_list = torch.jit.annotate(list[Tensor], [])
            retain_valweight_l_list = torch.jit.annotate(list[Tensor], [])

            weights_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())

                if model_name in ['IMV_full', 'IMV_tensor']:
                    val_alpha.append(unorm_s.detach().cpu().numpy())
                    val_beta.append(unorm_l.detach().cpu().numpy())
                if model_name in ['GLEN']:
                    #unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                    #valweight_s_list += [unorm_s]
                    unorm_l = unorm_l.squeeze(-1)  # shape time depth, batch, feature
                    valweight_l_list += [unorm_l]
                    #weights_list += [weights]

                if model_name in ['Retain']:
                    retain_valweight_l_list += [unorm_s]
        if model_name in ['GLEN']:
            valweight_l = torch.stack(valweight_l_list)
            valweight_l = valweight_l.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_l )
            valweight_l = valweight_l.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight_l = pd.DataFrame(valweight_l)

        if model_name in ['IMV_full', 'IMV_tensor']:
            val_alpha = np.concatenate(val_alpha)
            val_beta = np.concatenate(val_beta)
            val_alpha = val_alpha.mean(axis=0)
            val_beta = val_beta.mean(axis=0)
            val_beta = pd.DataFrame(val_beta)
            val_alpha = pd.DataFrame(val_alpha)
        if model_name in ['Retain']:
            retain_valweight = torch.stack(retain_valweight_l_list)
            retain_valweight = retain_valweight.reshape(-1, args.depth - 1, args.input_dim_s + args.input_dim_l)
            retain_valweight = retain_valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            retain_valweight = pd.DataFrame(retain_valweight)

        preds_val = np.concatenate(preds).reshape(-1)
        true_val = np.concatenate(true).reshape(-1)

        val_result = classification_report(true_val, preds_val, output_dict=True)
        val_result = pd.DataFrame(val_result).transpose()

        acc_val = accuracy_score(true_val, preds_val)
        pre_val = precision_score(true_val, preds_val, average='macro')
        recall_val = recall_score(true_val, preds_val, average='macro')
        f1_val = f1_score(true_val, preds_val, average='macro')
        kp_val = cohen_kappa_score(true_val, preds_val)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if i == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

        if acc_val > max_acc:
        #if recall_val > max_recall:
            max_acc = acc_val
            #max_recall = recall_val
            print("Saving...")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

            if model_name in ['IMV_full', 'IMV_tensor']:
                val_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_valalpha.csv'))
                val_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_valbeta.csv'))
            if model_name in['GLEN']:
                #valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_shop.csv'))
                valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_loan.csv'))
                #valweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_twoparts.csv'))

                soft_valweight_l = valweight_l.copy()

                for row in range(len(soft_valweight_l)):
                    soft_valweight_l.loc[row, :] = soft_valweight_l.loc[row, :].abs() / sum(
                        soft_valweight_l.loc[row, :].abs())

                soft_valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_valweight_loan.csv'))

            if model_name == 'Retain':
                retain_valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))

      #  if (i % 10 == 0):
           # print("lr: ", opt.param_groups[0]["lr"])
      #  print("Iter: ", i, "train: ", train_loss / (j), 'train_recall: ', recall_train, 'train_kp: ', kp_train,
           #   "val_recall: ", recall_val, 'val_kp', kp_val)

        if args.log:
            df_log_val.loc[i, 'Val Acc'] = acc_val
            df_log_val.loc[i, 'Val Precision'] = pre_val
            df_log_val.loc[i, 'Val Recall'] = recall_val
            df_log_val.loc[i, 'Val F-measure'] = f1_val
            df_log_val.loc[i, 'Val KP'] = kp_val

        epoch_scheduler.step(acc_val)

    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt')))
        preds = []
        true = []
        test_alpha = []
        test_beta = []

        #testweight_s_list = torch.jit.annotate(list[Tensor], [])
        testweight_l_list = torch.jit.annotate(list[Tensor], [])

        #weights_list = torch.jit.annotate(list[Tensor], [])
        # unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            # loss_test, unorm_weight,output,prob,_= model(batch_x.float(), batch_y,device)  # unorm time, batch, features
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            if model_name in ['IMV_full', 'IMV_tensor']:
                test_alpha.append(unorm_s.detach().cpu().numpy())
                test_beta.append(unorm_l.detach().cpu().numpy())
            if model_name in ['GLEN']:
                unorm_l = unorm_l.squeeze(-1)  # shape time depth, batch, feature
                testweight_l_list += [unorm_l]
            if model_name in ['Retain']:
                # shape time depth, batch, feature
                testweight_l_list += [unorm_s]

    preds_test = np.concatenate(preds).reshape(-1)
    true_test = np.concatenate(true).reshape(-1)
    test_result = classification_report(true_test, preds_test, output_dict=True)
    test_result = pd.DataFrame(test_result).transpose()

    acc_test = accuracy_score(true_test, preds_test)
    pre_test = precision_score(true_test, preds_test, average='macro')
    recall_test = recall_score(true_test, preds_test, average='macro')
    f1_test = f1_score(true_test, preds_test, average='macro')
    kp_test = cohen_kappa_score(true_test, preds_test)

    if model_name in ['Retain']:
        test_weight = torch.stack(testweight_l_list)  # shape 19, 5, 100, 5
        test_weight = test_weight.reshape(-1, args.depth - 1, args.input_dim_s+args.input_dim_l)
        test_weight = test_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        test_weight = pd.DataFrame(test_weight)
        test_weight.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight.csv'))
    if model_name in ['GLEN']:

        testweight_l = torch.stack(testweight_l_list)
        testweight_l = testweight_l.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_l )
        testweight_l = testweight_l.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweight_l = pd.DataFrame(testweight_l)
        testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_loan.csv'))

        soft_testweight_l = testweight_l.copy()
        for i in range(len(soft_testweight_l)):
            soft_testweight_l.loc[i, :] = soft_testweight_l.loc[i, :].abs() / sum(soft_testweight_l.loc[i, :].abs())

        soft_testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight_loan.csv'))

    if model_name in ['IMV_full', 'IMV_tensor']:
        test_alpha = np.concatenate(test_alpha)
        test_beta = np.concatenate(test_beta)
        test_alpha = test_alpha.mean(axis=0)
        test_beta = test_beta.mean(axis=0)
        test_beta = pd.DataFrame(test_beta)
        test_alpha = pd.DataFrame(test_alpha)
        test_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_testalpha.csv'))
        test_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_testbeta.csv'))

    df_log_test.loc[0, 'acc_test'] = acc_test
    df_log_test.loc[0, 'precision_test'] = pre_test
    df_log_test.loc[0, 'recall_test'] = recall_test
    df_log_test.loc[0, 'f1_test'] = f1_test
    df_log_test.loc[0, 'kp_test'] = kp_test


    print('acc_test, prec_test, recall_test,f1_test,kp_test',acc_test, pre_test, recall_test,f1_test,
          kp_test)
    time_final=time.time()-epoch_start_time

    outcome = np.vstack((true_test, preds_test))
    outcome = pd.DataFrame(outcome)
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path, str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    val_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'val_report_results.csv'))
    test_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'test_report_results.csv'))
    print('模型运行时间：{}'.format(time_final))

def credit_time_shop(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=20, verbose=True,
                                                                 min_lr=0.0000001)
    epochs =40
    max_acc = 0

    save_path = args.save_dirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        j = 0
        true_train = []
        preds_train = []
        for batch_x, batch_y in train_loader:
            j += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape batch, depth
            opt.zero_grad()

            loss_CE, output, _, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

            true_train.append(batch_y.detach().cpu().numpy())
            preds_train.append(output.detach().cpu().numpy())

            l = loss_CE
            l.backward()
            opt.step()
            train_loss += l.item()

        true_train = np.concatenate(true_train).reshape(-1)
        preds_train = np.concatenate(preds_train).reshape(-1)

        # 计算各种指标
        acc_train = accuracy_score(true_train, preds_train)
        pre_train = precision_score(true_train, preds_train, average='macro')
        recall_train = recall_score(true_train, preds_train, average='macro')
        f1_train = f1_score(true_train, preds_train, average='macro')
        kp_train = cohen_kappa_score(true_train, preds_train)

        # plot the roc curve for the model
        model_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        time_elapsed = time.time() - epoch_start_time
        #print(f"Training loss at epoch {i}: loss={train_loss / j:.4f} "
         #     f'kp_train :{kp_train: .2f}'
          #    f"time elapsed: {time_elapsed:.2f}")

        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train loss'] = train_loss / (j)
            df_log_val.loc[i, 'Train Acc'] = acc_train
            df_log_val.loc[i, 'Train Precision'] = pre_train
            df_log_val.loc[i, 'Train Recall'] = recall_train
            df_log_val.loc[i, 'Train F-measure'] = f1_train
            df_log_val.loc[i, 'Train KP'] = kp_train

        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            val_alpha = []
            val_beta = []
            valweight_s_list = torch.jit.annotate(list[Tensor], [])
            valweight_l_list = torch.jit.annotate(list[Tensor], [])
            retain_valweight_s_list = torch.jit.annotate(list[Tensor], [])

            weights_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())

                if model_name in ['IMV_full', 'IMV_tensor']:
                    val_alpha.append(unorm_s.detach().cpu().numpy())
                    val_beta.append(unorm_l.detach().cpu().numpy())
                if model_name in ['GLEN']:
                    unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                    valweight_s_list += [unorm_s]

                if model_name in ['Retain']:
                    retain_valweight_s_list += [unorm_s]
        if model_name in ['GLEN']:
            valweight_s = torch.stack(valweight_s_list)
            valweight_s = valweight_s.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_s )
            valweight_s = valweight_s.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight_s = pd.DataFrame(valweight_s)

        if model_name in ['IMV_full', 'IMV_tensor']:
            val_alpha = np.concatenate(val_alpha)
            val_beta = np.concatenate(val_beta)
            val_alpha = val_alpha.mean(axis=0)
            val_beta = val_beta.mean(axis=0)
            val_beta = pd.DataFrame(val_beta)
            val_alpha = pd.DataFrame(val_alpha)
        if model_name in ['Retain']:
            retain_valweight = torch.stack(retain_valweight_s_list)
            retain_valweight = retain_valweight.reshape(-1, args.depth - 1, args.input_dim_s + args.input_dim_l)
            retain_valweight = retain_valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            retain_valweight = pd.DataFrame(retain_valweight)
        preds_val = np.concatenate(preds).reshape(-1)
        true_val = np.concatenate(true).reshape(-1)

        val_result = classification_report(true_val, preds_val, output_dict=True)
        val_result = pd.DataFrame(val_result).transpose()

        acc_val = accuracy_score(true_val, preds_val)
        pre_val = precision_score(true_val, preds_val, average='macro')
        recall_val = recall_score(true_val, preds_val, average='macro')
        f1_val = f1_score(true_val, preds_val, average='macro')
        kp_val = cohen_kappa_score(true_val, preds_val)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if i == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

        if acc_val > max_acc:
        #if recall_val > max_recall:
            max_acc = acc_val
            #max_recall = recall_val
            print("Saving...")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

            if model_name in ['IMV_full', 'IMV_tensor']:
                val_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_valalpha.csv'))
                val_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_valbeta.csv'))
            if model_name in['GLEN']:
                valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_shop.csv'))
                #valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_loan.csv'))
                #valweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_twoparts.csv'))
                soft_valweight_s = valweight_s.copy()

                for row in range(len(soft_valweight_s)):
                    soft_valweight_s.loc[row, :] = soft_valweight_s.loc[row, :].abs() / sum(
                        soft_valweight_s.loc[row, :].abs())

                soft_valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_valweight_shop.csv'))

            if model_name == 'Retain':
                retain_valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))

       # if (i % 10 == 0):
          #  print("lr: ", opt.param_groups[0]["lr"])
       # print("Iter: ", i, "train: ", train_loss / (j), 'train_recall: ', recall_train, 'train_kp: ', kp_train,
        #      "val_recall: ", recall_val, 'val_kp', kp_val)

        if args.log:
            df_log_val.loc[i, 'Val Acc'] = acc_val
            df_log_val.loc[i, 'Val Precision'] = pre_val
            df_log_val.loc[i, 'Val Recall'] = recall_val
            df_log_val.loc[i, 'Val F-measure'] = f1_val
            df_log_val.loc[i, 'Val KP'] = kp_val

        epoch_scheduler.step(acc_val)

    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt')))
        preds = []
        true = []
        test_alpha = []
        test_beta = []

        testweight_s_list = torch.jit.annotate(list[Tensor], [])
        testweight_l_list = torch.jit.annotate(list[Tensor], [])

        weights_list = torch.jit.annotate(list[Tensor], [])
        # unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            # loss_test, unorm_weight,output,prob,_= model(batch_x.float(), batch_y,device)  # unorm time, batch, features
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            if model_name in ['IMV_full', 'IMV_tensor']:
                test_alpha.append(unorm_s.detach().cpu().numpy())
                test_beta.append(unorm_l.detach().cpu().numpy())
            if model_name in ['GLEN']:
                unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                testweight_s_list += [unorm_s]
            if model_name in ['Retain']:
                # shape time depth, batch, feature
                testweight_s_list += [unorm_s]

    preds_test = np.concatenate(preds).reshape(-1)
    true_test = np.concatenate(true).reshape(-1)
    test_result = classification_report(true_test, preds_test, output_dict=True)
    test_result = pd.DataFrame(test_result).transpose()

    acc_test = accuracy_score(true_test, preds_test)
    pre_test = precision_score(true_test, preds_test, average='macro')
    recall_test = recall_score(true_test, preds_test, average='macro')
    f1_test = f1_score(true_test, preds_test, average='macro')
    kp_test = cohen_kappa_score(true_test, preds_test)

    if model_name in ['Retain']:
        test_weight = torch.stack(testweight_s_list)  # shape 19, 5, 100, 5
        test_weight = test_weight.reshape(-1, args.depth - 1, args.input_dim_s)
        test_weight = test_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        test_weight = pd.DataFrame(test_weight)
        test_weight.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight.csv'))
    if model_name in ['GLEN']:
        testweight_s = torch.stack(testweight_s_list)
        testweight_s = testweight_s.permute(0, 2, 1, 3).reshape(-1, args.depth - 1, args.input_dim_s )
        testweight_s = testweight_s.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweight_s = pd.DataFrame(testweight_s)


        testweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_shop.csv'))
        #testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_loan.csv'))
        #testweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_twoparts.csv'))
        soft_testweight_s = testweight_s.copy()
        for i in range(len(soft_testweight_s)):
            soft_testweight_s.loc[i, :] = soft_testweight_s.loc[i, :].abs() / sum(soft_testweight_s.loc[i, :].abs())

        soft_testweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight_shop.csv'))


    if model_name in ['IMV_full', 'IMV_tensor']:
        test_alpha = np.concatenate(test_alpha)
        test_beta = np.concatenate(test_beta)
        test_alpha = test_alpha.mean(axis=0)
        test_beta = test_beta.mean(axis=0)
        test_beta = pd.DataFrame(test_beta)
        test_alpha = pd.DataFrame(test_alpha)
        test_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_testalpha.csv'))
        test_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_testbeta.csv'))

    df_log_test.loc[0, 'acc_test'] = acc_test
    df_log_test.loc[0, 'precision_test'] = pre_test
    df_log_test.loc[0, 'recall_test'] = recall_test
    df_log_test.loc[0, 'f1_test'] = f1_test
    df_log_test.loc[0, 'kp_test'] = kp_test


    print('acc_test, prec_test, recall_test,f1_test,kp_test',acc_test, pre_test, recall_test,f1_test,
          kp_test)
    time_final=time.time()-epoch_start_time


    outcome = np.vstack((true_test, preds_test))
    outcome = pd.DataFrame(outcome)
    #
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path, str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    val_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'val_report_results.csv'))
    test_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'test_report_results.csv'))
    print('模型运行时间：.{}'.format(time_final))

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

##非时序
def credit_notime(model, model_name,  X_train_t,y_train_t,X_test_t,y_test_t, args, device, exp_id):

    save_path = args.save_dirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    epoch_start_time = time.time()
    loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(X_train_t,y_train_t,X_test_t,y_test_t,  device)

    preds_test = output.reshape(-1)
    true_test = y_test_t.reshape(-1)
    test_result = classification_report(true_test, preds_test, output_dict=True)
    test_result = pd.DataFrame(test_result).transpose()

    acc_test = accuracy_score(true_test, preds_test)
    pre_test = precision_score(true_test, preds_test, average='macro')
    recall_test = recall_score(true_test, preds_test, average='macro')
    f1_test = f1_score(true_test, preds_test, average='macro')
    kp_test=cohen_kappa_score(true_test,preds_test)

    df_log_test.loc[0, 'acc_test'] = acc_test
    df_log_test.loc[0, 'precision_test'] = pre_test
    df_log_test.loc[0, 'recall_test'] = recall_test
    df_log_test.loc[0, 'f1_test'] = f1_test
    df_log_test.loc[0, 'kp_test'] = kp_test

    print('acc_test,prec_test,recal_test,f1_test,kp_test',  acc_test, pre_test,recall_test,f1_test,kp_test)
    time_elapsed = time.time() - epoch_start_time

    outcome = np.vstack((true_test, preds_test))
    outcome = pd.DataFrame(outcome)
    model_save_path = os.path.join(save_path, model_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    test_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'test_report_results.csv'))
    print('模型训练时间：{}'.format(time_elapsed))

def lrp_new_credit_training2(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id,X_train_t,X_test_t):
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=20, verbose=True,
                                                                 min_lr=0.0000001)
    epochs = 1
    max_acc = 0

    save_path = args.save_dirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    start_time=time.time()
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        j = 0
        true_train = []
        prob_list_train = []
        preds_train = []
        for batch_x, batch_y in train_loader:
            j += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape batch, depth
            opt.zero_grad()
            loss_per=0
            weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
            pred_list = torch.jit.annotate(list[Tensor], [])
            criterion=torch.nn.CrossEntropyLoss(weight=weights)
            for number in range(1,8):
                prob = model.train_forward(batch_x[:,:number].float()) #shape batch,5
                loss_per=loss_per+criterion(prob,batch_y[:,number-1].long())
                prediction=torch.argmax(prob,dim=1)
                ##shape prediction batch,
                pred_list+=[prediction]
            pred = torch.stack(pred_list).permute(1, 0)
            true_train.append(batch_y.detach().cpu().numpy())
            preds_train.append(pred.detach().cpu().numpy())

            l = loss_per
            l.backward()
            opt.step()
            train_loss += l.item()

        true_train = np.concatenate(true_train).reshape(-1)
        preds_train = np.concatenate(preds_train).reshape(-1)

        #计算指标
        acc_train = accuracy_score(true_train, preds_train)
        pre_train = precision_score(true_train, preds_train, average='macro')
        recall_train = recall_score(true_train, preds_train, average='macro')
        f1_train = f1_score(true_train, preds_train, average='macro')
        kp_train = cohen_kappa_score(true_train, preds_train)
        model_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train loss'] = train_loss / (j)
            df_log_val.loc[i, 'Train Acc'] = acc_train
            df_log_val.loc[i, 'Train Precision'] = pre_train
            df_log_val.loc[i, 'Train Recall'] = recall_train
            df_log_val.loc[i, 'Train F-measure'] = f1_train
            df_log_val.loc[i, 'Train KP'] = kp_train

        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss_per = 0
                weights = torch.FloatTensor([3.44, 5.52, 1.89]).cuda(1)
                pred_list = torch.jit.annotate(list[Tensor], [])
                #pred_list=[]
                criterion = torch.nn.CrossEntropyLoss(weight=weights)
                for number in range(1, 8):
                    prob = model.train_forward(batch_x[:, :number].float())
                    # prob_time=prob[:,number,:] #shape batch, 5
                    loss_per = loss_per + criterion(prob, batch_y[:, number - 1].long())
                    prediction = torch.argmax(prob, dim=1)
                    pred_list += [prediction]
                pred= torch.stack(pred_list).permute(1, 0)
                true.append(batch_y.detach().cpu().numpy())
                preds.append(pred.detach().cpu().numpy())
        preds_val = np.concatenate(preds).reshape(-1)
        true_val = np.concatenate(true).reshape(-1)

        val_result = classification_report(true_val, preds_val, output_dict=True)
        val_result = pd.DataFrame(val_result).transpose()

        acc_val = accuracy_score(true_val, preds_val)
        pre_val = precision_score(true_val, preds_val, average='macro')
        recall_val = recall_score(true_val, preds_val, average='macro')
        f1_val = f1_score(true_val, preds_val, average='macro')
        kp_val = cohen_kappa_score(true_val, preds_val)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if i==0:
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

        if acc_val > max_acc:
            max_acc = acc_val
            print("Saving...")
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))

        if args.log:
            df_log_val.loc[i, 'Val Acc'] = acc_val
            df_log_val.loc[i, 'Val Precision'] = pre_val
            df_log_val.loc[i, 'Val Recall'] = recall_val
            df_log_val.loc[i, 'Val F-measure'] = f1_val
            df_log_val.loc[i, 'Val kp'] = kp_val

        epoch_scheduler.step(acc_val)

    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt')))
        eps=0.001
        bias_factor=0.0
        rx_list_all=[]
        for i in range(X_test_t.shape[0]):
        #for i in range(2):
            rx_sample=[] #是一个list，每个list有32个元素
            for number in range(1,8):
                model.set_input(X_test_t[i][:number])
                prob=model.forward()
                prediction = np.argmax(prob)
                target_class=prediction
                Rx, R_rest = model.lrp(X_test_t[i][:number], target_class, eps, bias_factor)
                rx_sample.append(np.sum(Rx,axis=0))#shape 32
            rx_sample=np.vstack(rx_sample)
            rx_list_all.append(rx_sample)
        new_rx_list_all=np.stack(rx_list_all)
        new_rx_list_all=np.mean(new_rx_list_all,axis=0)
        time_lrp=time.time()
        print('LRP 运行时间：{}'.format(time_lrp - start_time))

    df_log_val.to_csv(os.path.join(model_save_path, str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    val_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'val_report_results.csv'))
   # new_rx_list_all.to_csv(os.path.join(model_save_path,str(exp_id)+'lrp_result.csv'))
    #rx_list_all.to_csv(os.path.join(model_save_path,str(exp_id)+'all_lrp_result.csv'))
    save_path =os.path.join(model_save_path,str(exp_id)+'lrp_result.csv')
    np.savetxt(save_path, new_rx_list_all, delimiter=',')
    save_path = os.path.join(model_save_path, str(exp_id) + 'all_lrp_result.npy')
    np.save(save_path, rx_list_all)


def new_credit_time(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=20, verbose=True,
                                                                 min_lr=0.0000001)
    epochs = 40
    max_acc = 0

    save_path = args.save_dirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        j = 0
        true_train = []
        preds_train = []
        for batch_x, batch_y in train_loader:
            j += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape batch, depth
            opt.zero_grad()

            loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            true_train.append(batch_y.detach().cpu().numpy())
            preds_train.append(output.detach().cpu().numpy())

            l = loss_CE
            l.backward()
            opt.step()
            train_loss += l.item()
            #print(l.item())

        true_train = np.concatenate(true_train).reshape(-1)
        preds_train = np.concatenate(preds_train).reshape(-1)

        #计算指标
        acc_train = accuracy_score(true_train, preds_train)
        pre_train = precision_score(true_train, preds_train, average='macro')
        recall_train = recall_score(true_train, preds_train, average='macro')
        f1_train = f1_score(true_train, preds_train, average='macro')
        kp_train = cohen_kappa_score(true_train, preds_train)

        model_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        time_elapsed = time.time() - epoch_start_time
        #print(f"Training loss at epoch {i}: loss={train_loss / j:.4f} "
         #f'kp_train :{kp_train: .2f}'
          #   f"time elapsed: {time_elapsed:.2f}")

        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train loss'] = train_loss / (j)
            df_log_val.loc[i, 'Train Acc'] = acc_train
            df_log_val.loc[i, 'Train Precision'] = pre_train
            df_log_val.loc[i, 'Train Recall'] = recall_train
            df_log_val.loc[i, 'Train F-measure'] = f1_train
            df_log_val.loc[i, 'Train KP'] = kp_train

        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            val_alpha = []
            val_beta = []
            valweight_s_list = torch.jit.annotate(list[Tensor], [])
            valweight_l_list = torch.jit.annotate(list[Tensor], [])
            retain_valweight_s_list = torch.jit.annotate(list[Tensor], [])

            weights_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)

                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())

                if model_name in ['IMV_full', 'IMV_tensor']:
                    val_alpha.append(unorm_s.detach().cpu().numpy())
                    val_beta.append(unorm_l.detach().cpu().numpy())
                if model_name in ['GLEN']:
                    unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                    valweight_s_list += [unorm_s]
                    unorm_l = unorm_l.squeeze(-1)  # shape time depth, batch, feature
                    valweight_l_list += [unorm_l]
                    weights_list += [weights]

                if model_name in ['Retain']:
                    retain_valweight_s_list += [unorm_s]
        if model_name in ['GLEN']:
            valweight_s = torch.stack(valweight_s_list)
            valweight_s = valweight_s.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim_s )
            valweight_s = valweight_s.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight_s = pd.DataFrame(valweight_s)

            valweight_l = torch.stack(valweight_l_list)
            valweight_l = valweight_l.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim_l )
            valweight_l = valweight_l.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight_l = pd.DataFrame(valweight_l)

            valweights = torch.stack(weights_list)
            valweights = valweights.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweights = pd.DataFrame(valweights)

        if model_name in ['IMV_full', 'IMV_tensor']:
            val_alpha = np.concatenate(val_alpha)
            val_beta = np.concatenate(val_beta)
            val_alpha = val_alpha.mean(axis=0)
            val_beta = val_beta.mean(axis=0)
            val_beta = pd.DataFrame(val_beta)
            val_alpha = pd.DataFrame(val_alpha)
        if model_name in ['Retain']:
            retain_valweight = torch.stack(retain_valweight_s_list)
            retain_valweight = retain_valweight.reshape(-1, args.depth - 2, args.input_dim_s + args.input_dim_l)
            retain_valweight = retain_valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            retain_valweight = pd.DataFrame(retain_valweight)
        preds_val = np.concatenate(preds).reshape(-1)
        true_val = np.concatenate(true).reshape(-1)
        val_result = classification_report(true_val, preds_val, output_dict=True)
        val_result = pd.DataFrame(val_result).transpose()

        acc_val = accuracy_score(true_val, preds_val)
        pre_val = precision_score(true_val, preds_val, average='macro')
        recall_val = recall_score(true_val, preds_val, average='macro')
        f1_val = f1_score(true_val, preds_val, average='macro')
        kp_val = cohen_kappa_score(true_val, preds_val)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if i==0:
            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))
        if acc_val > max_acc:
            max_acc = acc_val
            print("Saving...")

            torch.save(model.state_dict(), os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt'))
            if model_name in ['IMV_full', 'IMV_tensor']:
                val_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_valalpha.csv'))
                val_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_valbeta.csv'))
            if model_name in ['GLEN']:
                valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_shop.csv'))
                valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_loan.csv'))
                valweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight_twoparts.csv'))
                soft_valweight_s = valweight_s.copy()

                for row in range(len(soft_valweight_s)):
                    soft_valweight_s.loc[row, :] = soft_valweight_s.loc[row, :].abs() / sum(
                        soft_valweight_s.loc[row, :].abs())

                soft_valweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_valweight_shop.csv'))
                soft_valweight_l = valweight_l.copy()

                for row in range(len(soft_valweight_l)):
                    soft_valweight_l.loc[row, :] = soft_valweight_l.loc[row, :].abs() / sum(
                        soft_valweight_l.loc[row, :].abs())

                soft_valweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_valweight_loan.csv'))

            if model_name == 'Retain':
                retain_valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))

    #    if (i % 10 == 0):
      #      print("lr: ", opt.param_groups[0]["lr"])
    #    print("Iter: ", i, "train: ", train_loss / (j), 'train_recall: ', recall_train, 'train_kp: ', kp_train,
       #       "val_recall: ", recall_val, 'val_kp', kp_val)

        if args.log:
            df_log_val.loc[i, 'Val Acc'] = acc_val
            df_log_val.loc[i, 'Val Precision'] = pre_val
            df_log_val.loc[i, 'Val Recall'] = recall_val
            df_log_val.loc[i, 'Val F-measure'] = f1_val
            df_log_val.loc[i, 'Val kp'] = kp_val

        epoch_scheduler.step(acc_val)

    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path, str(exp_id) + '_credit_predict.pt')))
        preds = []
        true = []
        test_alpha = []
        test_beta = []

        testweight_s_list = torch.jit.annotate(list[Tensor], [])
        testweight_l_list = torch.jit.annotate(list[Tensor], [])

        retain_testweight_s_list = torch.jit.annotate(list[Tensor], [])

        weights_list = torch.jit.annotate(list[Tensor], [])
        # unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss_CE, output, prob, unorm_s, unorm_l, weights, _ = model(batch_x.float(), batch_y, device)
            # loss_test, unorm_weight,output,prob,_= model(batch_x.float(), batch_y,device)  # unorm time, batch, features
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())

            if model_name in ['IMV_full', 'IMV_tensor']:
                test_alpha.append(unorm_s.detach().cpu().numpy())
                test_beta.append(unorm_l.detach().cpu().numpy())
            if model_name in ['GLEN']:
                unorm_s = unorm_s.squeeze(-1)  # shape time depth, batch, feature
                testweight_s_list += [unorm_s]
                unorm_l = unorm_l.squeeze(-1)  # shape time depth, batch, feature
                testweight_l_list += [unorm_l]
                weights_list += [weights]

            if model_name in ['Retain']:
                # shape time depth, batch, feature
                retain_testweight_s_list += [unorm_s]

    preds_test = np.concatenate(preds).reshape(-1)
    true_test = np.concatenate(true).reshape(-1)
    test_result = classification_report(true_test, preds_test, output_dict=True)
    test_result = pd.DataFrame(test_result).transpose()

    acc_test = accuracy_score(true_test, preds_test)
    pre_test = precision_score(true_test, preds_test, average='macro')
    recall_test = recall_score(true_test, preds_test, average='macro')
    f1_test = f1_score(true_test, preds_test, average='macro')
    kp_test = cohen_kappa_score(true_test, preds_test)

    if model_name in ['Retain']:
        retain_test_weight = torch.stack(retain_testweight_s_list)  # shape 19, 5, 100, 5
        retain_test_weight = retain_test_weight.reshape(-1, args.depth - 2, args.input_dim_s + args.input_dim_l)
        retain_test_weight = retain_test_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        retain_test_weight = pd.DataFrame(retain_test_weight)
        retain_test_weight.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight.csv'))
    if model_name in ['GLEN']:
        testweight_s = torch.stack(testweight_s_list)
        testweight_s = testweight_s.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim_s )
        testweight_s = testweight_s.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweight_s = pd.DataFrame(testweight_s)

        testweight_l = torch.stack(testweight_l_list)
        testweight_l = testweight_l.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim_l)
        testweight_l = testweight_l.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweight_l = pd.DataFrame(testweight_l)

        testweights = torch.stack(weights_list)
        testweights = testweights.mean(dim=0, keepdim=False).detach().cpu().numpy()
        testweights = pd.DataFrame(testweights)

        testweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_shop.csv'))
        testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_loan.csv'))
        testweights.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight_twoparts.csv'))
        soft_testweight_s = testweight_s.copy()
        for i in range(len(soft_testweight_s)):
            soft_testweight_s.loc[i, :] = soft_testweight_s.loc[i, :].abs() / sum(soft_testweight_s.loc[i, :].abs())

        soft_testweight_s.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight_shop.csv'))

        soft_testweight_l = testweight_l.copy()
        for i in range(len(soft_testweight_l)):
            soft_testweight_l.loc[i, :] = soft_testweight_l.loc[i, :].abs() / sum(soft_testweight_l.loc[i, :].abs())

        soft_testweight_l.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight_loan.csv'))

    if model_name in ['IMV_full', 'IMV_tensor']:
        test_alpha = np.concatenate(test_alpha)
        test_beta = np.concatenate(test_beta)
        test_alpha = test_alpha.mean(axis=0)
        test_beta = test_beta.mean(axis=0)
        test_beta = pd.DataFrame(test_beta)
        test_alpha = pd.DataFrame(test_alpha)
        test_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_testalpha.csv'))
        test_beta.to_csv(os.path.join(model_save_path, str(exp_id) + '_testbeta.csv'))

    df_log_test.loc[0, 'acc_test'] = acc_test
    df_log_test.loc[0, 'precision_test'] = pre_test
    df_log_test.loc[0, 'recall_test'] = recall_test
    df_log_test.loc[0, 'f1_test'] = f1_test
    df_log_test.loc[0, 'k_test'] = kp_test

    print('acc_test, prec_test, recall_test,f1_test,kp_test',acc_test, pre_test, recall_test,f1_test,
          kp_test)

    outcome = np.vstack((true_test, preds_test))
    outcome = pd.DataFrame(outcome)
    time_final=time.time()-epoch_start_time
    #
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path, str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))
    val_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'val_report_results.csv'))
    test_result.to_csv(os.path.join(model_save_path, str(exp_id) + 'test_report_results.csv'))
    print('模型训练时间：{}'.format(time_elapsed))
    print('模型训运行间：{}'.format(time_final))


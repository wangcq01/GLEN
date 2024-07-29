
def config_credit(model_name, args):
    params = {}
    # some common params for all models
    params['input_dim_s'] = args.input_dim_s
    params['input_dim_l'] = args.input_dim_l
    params['n_units'] = args.n_units
    params['bias'] = True
    params['time_depth']=args.depth
    params['output_dim']=args.output_dim

    if model_name in ['GLEN']:
        params['N_units'] = args.n_units
        params['num_classes'] = args.num_classes

    elif model_name == 'IMV_full':
            params['n_units']=8
            params['input_dim'] = args.input_dim_s + args.input_dim_l

    elif model_name == 'IMV_tensor':
        params['input_dim'] = args.input_dim_s + args.input_dim_l

    elif model_name == 'Retain':

        params['inputDimSize'] = args.input_dim_s+args.input_dim_l
        params['embDimSize'] = 8
        params['alphaHiddenDimSize']=64
        params['betaHiddenDimSize'] =32
        params['outputDimSize']=args.output_dim
        params['keep_prob']=1.0

    elif model_name == 'LSTM':
        params['input_dim']=args.input_dim_s+args.input_dim_l
        params['num_classes']=args.num_classes
        pass

    elif model_name in ['GRU','SHAP','LRP']:
        params['input_dim']=args.input_dim_s+args.input_dim_l
        params['num_classes'] = args.num_classes
        pass
    elif model_name=='MLP':
        params['output_dim'] = args.num_classes
        params['layer_size'] = args.n_units
    elif model_name in ['LR', 'SVM','RF','LDA','NB']:
      #  params['input_dim']=args.input_dim_s+args.input_dim_l
        pass
    else:
        raise ModuleNotFoundError

    return params

def config(model_name, args):
    params = config_credit(model_name, args)

    return params


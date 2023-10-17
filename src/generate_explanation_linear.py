
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import pickle as pkl
from numpy import linalg as LA
from utils import convert_sklearn_to_torch_model
import torch.nn as nn
## Dataset Prep
import pandas as pd
import torch.nn as nn
from torchvision import  datasets, transforms
from torch.utils.data import DataLoader
from model import LR_Model
import ML_Models.data_loader as loader
from torch.autograd.functional import hessian
import numpy as np
from torch.nn import Module
import torch.nn.functional as F
from recourse_methods import RobustRecourse, CCHVAE, SCFE_modified

parser = argparse.ArgumentParser(
                    prog = 'LR Explanation',
                    description = 'Explanations')

parser.add_argument('-f', '--file_number')      
parser.add_argument('-d', '--data_name')      


def main(file_number, data_name):

    n_starting_instances = 5000

    #German Dataset
    german_dict = {
        "data_path": '../Data_Sets/German_Credit_Data/',
        "filename_train": 'german-train.csv',
        "filename_test": 'german-test.csv',
        "label": "credit-risk",
        "task": "classification",
        "continuous_feats" : ['account_bal', 'duration', 'payment_status', 'credit_amount','savings_bond_value', 'employed_since', 'intallment_rate', 'residence_since', 'age', 'number_of_existcr', 'job','number_of_dependents'],
        "binary_feats" : [],
        "lr": 1e-3,
        "d": 6,
        "H1": 24,
        "H2": 48,
        "activFun": nn.Softplus(),
        "n_starting_instances": 200
    
    }
    
    # Adult Dataset 
    adult_dict = {
        "data_path": "../Data_Sets/Adult/",
        "filename_train": 'adult-train.csv',
        "filename_test": 'adult-test.csv',
        #"filename_test": 'adult-test-{}.csv'.format(file_number),
        "label": 'income',
        "task": "classification",
        "continuous_feats" : ["age","fnlwgt","education-num", "capital-gain", "capital-loss" ,"hours-per-week"],
        "binary_feats" : ["sex_Male", "workclass_Private", "marital-status_Non-Married", "occupation_Other" ,"relationship_Non-Husband", "race_White", "native-country_US"],
        "lr": 1e-3,
        "d": 6,
        "H1": 26,
        "H2": 52,
        "activFun": nn.Softplus(),
        "n_starting_instances": 9045
    }
    
    compas_dict = {
        "data_path": '../Data_Sets/COMPAS/',
        "filename_train": 'compas-train.csv',
        # "filename_test": 'compas-test.csv',
        "filename_test": 'compas-test-{}.csv'.format(file_number),
        "label": "two_year_recid",
        "task": "classification",
        "lr": 1e-3,
        "d": 6,
        "H1": 14,
        "H2": 28,
        "activFun": nn.Softplus(),
        "batch_size": 32,
        "n_starting_instances": 1235,
        'lambda_reg': 1e-6,
        "epochs": 100
    }


    

    data_meta_dictionaries = {
            "adult": adult_dict,
            "compas": compas_dict, 
            "german": german_dict
        }
    data_meta_info = data_meta_dictionaries[data_name]


    dataset_test = loader.DataLoader_Tabular(path=data_meta_info["data_path"],
                                                     filename=data_meta_info["filename_test"],
                                                     label=data_meta_info["label"], scale = "standard")

    dataset_train = loader.DataLoader_Tabular(path=data_meta_info["data_path"],
                                                      filename=data_meta_info["filename_train"],
                                                      label=data_meta_info["label"], scale = "standard")


    column_names = pd.read_csv(data_meta_info["data_path"] + data_meta_info["filename_train"]).drop(data_meta_info["label"], axis=1).columns

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




    train_loader = DataLoader(dataset_train, batch_size = len(dataset_train), shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size = len(dataset_test), shuffle=False)

    for X,y,ind in train_loader:
        print("Train size : ", X.size(), y.size(), ind.size())


    for X_test,y_test,ind in test_loader:
        print("Test size : ", X_test.size(), y_test.size(), ind.size())

    num_input = X.size()[1]
    total_train_samples = X.size()[0]
    total_test_samples = X_test.size()[0]

    clf = LogisticRegression(random_state=0, C = 100).fit(X.numpy(), y.numpy())
    w = torch.cat([torch.from_numpy(clf.coef_).squeeze()])
    model_D = convert_sklearn_to_torch_model(clf, num_input, device)
    path_to_store_model = "./models/{}_lr_model_D.pth".format(data_name)
    torch.save(model_D, path_to_store_model)



    ## LKO Approximation : Find G_i and H
 
    model_D = torch.load("./models/{}_lr_model_D.pth".format(data_name), map_location=torch.device('cpu')).to(device) #cpu()
    w = torch.cat([torch.clone(model_D.linear.weight.data).squeeze(), torch.clone(model_D.linear.bias.data)]).unsqueeze(0).T
    opt = optim.SGD(model_D.parameters(), lr=1e-3)
    g_is = []
    train_loader = DataLoader(dataset_train, batch_size = 1, shuffle=True)

    # H_avg = None
    for X,y,ind in train_loader:
        X = X.to(device)
        y = y.to(device)
        #print(y)
        yp = model_D(X.view(X.shape[0], -1).to(torch.float32).to(device))[:,0]
        # print(yp, clf.decision_function(X)) # passes this test 
        loss = torch.nn.functional.binary_cross_entropy_with_logits(yp, y.float())
        opt.zero_grad()
        loss.backward()
        g_is.append(torch.cat([torch.clone(model_D.linear.weight.grad).squeeze(), torch.clone(model_D.linear.bias.grad)]))
        opt.zero_grad()
    g_is_tensored = torch.stack(g_is)





    # Computing H_inverse
    from torch.autograd.functional import hessian
    train_loader = DataLoader(dataset_train, batch_size = len(dataset_train), shuffle=False)
    for X,y,ind in train_loader:
        X1 = torch.cat([X, torch.ones((X.size(0), 1))], dim=1)
        #print(X1.size(), y.size(), ind.size())
    w = w.squeeze()
    def loss(w):
        logits = torch.matmul(X1.float().to(device), w.float().to(device))
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(dtype=torch.float32).to(device))

    H = hessian(loss, w)
    print(H.size())

    hess = H.cpu().numpy()
    hess = (hess + hess.T) / 2

    ew, ev = np.linalg.eigh(hess)
    print(ew)

    print("min abs eigenvalues", np.abs(ew).min())
    print("X rank", np.linalg.matrix_rank(X))

    hess_pinv = np.linalg.pinv(hess)
    print(np.linalg.norm(hess_pinv))

    hess_reg_inv = np.linalg.inv(hess + 0.01 * np.eye(hess.shape[0]))
    H_inverse = torch.tensor(hess_reg_inv)
    H_g_prod = torch.matmul(g_is_tensored.float().to(device), H_inverse.float().to(device))

    print(np.linalg.norm(hess_reg_inv), np.linalg.norm(hess))


    # A set
    def get_A(x, H_g_prod):
        return torch.matmul(H_g_prod.float().to(device), torch.cat([x.float().to(device), torch.ones((1, x.size(1))).to(device)], dim=0)).squeeze()

    def get_k_min_values(A, m, k, gs_tensored, H_inverse):
        if m == 0:
            indices =  (A < torch.kthvalue(A, k+1).values)
        else:
            indices =  (A >= torch.kthvalue(A, m).values) and (A < torch.kthvalue(A, m+k+1).values)
        gs_filtered = torch.clone(gs_tensored[indices])
        return torch.sum(torch.matmul(gs_filtered.float().to(device), H_inverse.float().to(device)), dim = 0).float()/total_train_samples


    norm = 2
    test_loader = DataLoader(dataset_test, batch_size = len(dataset_test), shuffle=False)
    for X,y,ind in test_loader:
        yp = model_D(X.view(X.shape[0], -1).to(torch.float32))[:,0]
    total_invalid_samples = sum(yp < 0).item()
    print("Total invalid samples : ", total_invalid_samples)

    # Compute cost and validity of counterfactual generated robust to data removal
    import Recourse_Methods.Generative_Model.model as model_vae

    test_loader = DataLoader(dataset_test, batch_size = 1, shuffle=False)
    model_D = torch.load("./models/{}_lr_model_D.pth".format(data_name), map_location=torch.device('cpu')).to(device)
    counter = 0
    K_percentages = [0, 0.5, 1 , 2, 3 , 5] #, 20, 25, 30]
    k_vals = {i : int((i * total_train_samples)/100) for i in K_percentages}
    explanation_methods = ["C-CHVAE"] +  ["RR-CFE-{}".format(i) for i in K_percentages] + ["ROAR-0.1"] #, "ROAR-0.05", "ROAR-0.02"]

    cfe_test_set  = dict()
    for explanation_method in explanation_methods:
        cfe_test_set[explanation_method] = []
        
    # Find lambda for ROAR
    def get_lambda_val(recourse_cls, model_D, lamb_iter = 0.0 , step = 0.1, val_samples = 5):
        count_valid = val_samples
        print("Total validation samples :", val_samples)
        while count_valid == val_samples:
            print("Lambda Values : ", lamb_iter)
            test_loader = DataLoader(dataset_test, batch_size = 1, shuffle=False)
            count_valid = 0
            count_samples = 0
            for X,y,ind in test_loader:
                yp = model_D(X.view(X.shape[0], -1).to(torch.float32))[:,0].item()
                if yp < 0:
                    count_samples += 1
                    rfe_rr, delta_W_0_005,_  = robust_recourse.get_recourse(X.T.cpu().numpy(), lamb=lamb_iter)
                    if model_D(rfe_rr.squeeze().view(X.shape[0], -1).to(torch.float32))[:,0].item() > 0:
                        count_valid += 1

                if count_samples == val_samples:
                    break
            if count_valid == val_samples:
                lamb_iter += step

        return lamb_iter-step
    
    lamb_iters = dict()
    test_loader = DataLoader(dataset_test, batch_size = len(dataset_test), shuffle=False)
    for X,y,ind in test_loader:
        yp = model_D(X.view(X.shape[0], -1).to(torch.float32))[:,0]

    total_samples_samples = sum(yp < 0).item()
    coefficients = model_D.linear.weight.data
    bias = model_D.linear.bias.data
    deltas = [0.1] 
    for delta_val in deltas:
        print("Processing {}...".format(delta_val))
        robust_recourse = RobustRecourse(W=torch.clone(coefficients).squeeze().numpy(), W0 = torch.clone(bias).numpy(), feature_costs=None, delta_max = delta_val, norm = norm)
        lamb_iters[delta_val] = get_lambda_val(robust_recourse, model_D, val_samples = max([int(total_samples_samples*0.05), 5]))

    pkl.dump(lamb_iters, open("./hyperparameters/{}_hyper_params_delta_{}_ROAR_lr.pkl".format(data_name, norm), "wb"))
    
    
    lamb_iters = pkl.load(open("./hyperparameters/{}_hyper_params_delta_{}_ROAR_lr.pkl".format(data_name, norm), "rb"))
    
    num_negative_samples = 0
    target_delta = 0.05
    from datetime import datetime
    test_loader = DataLoader(dataset_test, batch_size = 1, shuffle=False)
    for X,y,ind in test_loader:
        start_time = datetime.now() 
        X = X.to(device)
        y = y.to(device)
        model_D = model_D.to(device)
        yp = model_D(X.view(X.shape[0], -1).to(torch.float32))[:,0].item()
        if yp < 0:
            print("Processing {} sample".format(ind))
            num_negative_samples += 1
            for k_val in k_vals.keys():
                print("processing .. ", k_val)
                A_m = get_A(X.T.to(device), H_g_prod.to(device))
                fmk_params = torch.cat([model_D.linear.weight.data.squeeze(), model_D.linear.bias.data]) + get_k_min_values(A_m, 0, k_vals[k_val], g_is_tensored, H_inverse)
                model_appx = LR_Model(num_input).to(device)
                model_appx.linear.weight.data = fmk_params[:-1].unsqueeze(0)
                model_appx.linear.bias.data = fmk_params[-1]
                rr_cfe = SCFE_modified(model_appx, _lambda = 2.5, step=2, max_iter = 100000, lr = 0.5 , target_threshold = 0, norm = norm)
                cfe_rr, cfe_distance_rr = rr_cfe.generate_counterfactuals(X.float())
                cfe_test_set["RR-CFE-{}".format(k_val)].append([cfe_rr.squeeze().cpu().numpy(), torch.norm(cfe_rr.squeeze().to(device) - X.float().squeeze().to(device), p = norm).cpu().item(), X.squeeze().float()])

            # C-CHVAE
            vae_path = "../Recourse_Methods/Generative_Model/Saved_Models/"
            input_size = dataset_train.get_number_of_features()
            vae_model = model_vae.VAE_model(input_size,
                                    data_meta_info['activFun'],
                                    data_meta_info['d'],
                                    data_meta_info['H1'],
                                    data_meta_info['H2'])

            data_meta_info["vae_path"] = vae_path + f"vae_{data_name}_rr_cfe.pt"
            vae_model.load_state_dict(torch.load(data_meta_info["vae_path"]))
            cchvae = CCHVAE(classifier=model_D, model_vae=vae_model, n_search_samples = 1000, step=0.05, max_iter=10000, target_threshold=0, p_norm = norm)
            ccvae_cfe, ccvae_val,  = cchvae.generate_counterfactuals(
                    query_instance= X.float().reshape(1, -1),
                    target_class= 1 )
            cfe_test_set["C-CHVAE"].append([ccvae_cfe.squeeze().numpy(), torch.norm(ccvae_cfe.squeeze().to(device) - X.float().squeeze().to(device), p = norm).cpu().item(), X.squeeze().float()])

            # ROAR
            coefficients = model_D.linear.weight.data
            bias = model_D.linear.bias.data
            for delta_val in deltas: 
                robust_recourse = RobustRecourse(W=torch.clone(coefficients).squeeze().numpy(), W0 = torch.clone(bias).numpy(), feature_costs=None, delta_max = delta_val, norm = norm)
                roar_cfe, delta_W,_  = robust_recourse.get_recourse(X.T.cpu().numpy(), lamb=lamb_iters[delta_val])
                cfe_test_set["ROAR-{}".format(delta_val)].append([roar_cfe.squeeze().numpy(), torch.norm(roar_cfe.squeeze() - X.cpu().float().squeeze(), p = norm).item(), X.squeeze().float()])

            
            if num_negative_samples%100 == 0:
                pkl.dump(cfe_test_set, open( "./explanations/cost_cfe_dict_explanation_methods_{}_hps_{}_file_name_{}_lr.pkl".format(data_name, "_".join([str(i) for i in K_percentages]), str(file_number)), "wb"))



    pkl.dump(cfe_test_set, open( "./explanations/cost_cfe_dict_explanation_methods_{}_hps_{}_file_name_{}_lr.pkl".format(data_name, "_".join([str(i) for i in K_percentages]),  str(file_number)), "wb"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.file_number, args.data_name)

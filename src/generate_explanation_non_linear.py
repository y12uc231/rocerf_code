


# please update this with your project directory path
import sys
path_local = "/Users/<local>/icml_sub"

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
from utils import convert_sklearn_to_torch_model, convert_sklearn_to_torch_model_v2
import torch.nn as nn
import pickle as pkl
## Dataset Prep
import pandas as pd
import torch.nn as nn
from torchvision import  datasets, transforms
from torch.utils.data import DataLoader
from model import LR_Model, NN3_ModelSP
import ML_Models.data_loader as loader
from captum.attr import LimeBase
from torch import Tensor
import Recourse_Methods.Generative_Model.model as model_vae
import torch.optim as optim
from recourse_methods import RobustRecourse, CCHVAE , SCFE_modified 
from captum._utils.models.linear_model import SkLearnLinearModel, SGDLinearModel
from torch.autograd.functional import hessian
import torch.nn.functional as F

parser = argparse.ArgumentParser(
                    prog = 'Adult Explanation',
                    description = 'Explanations')

parser.add_argument('-f', '--file_number')      
parser.add_argument('-d', '--data_name')      
parser.add_argument('-li', '--is_lime')      
parser.add_argument('-lr', '--find_lambda_roar')      
parser.add_argument('-b', '--beta')      

def main(file_number, data_name, beta = 1, is_lime = False, find_lambda_roar = False ):
    n_layers = 3
    norm = 2
    n_starting_instances = 5000
    import pickle as pkl
    #German Dataset
    german_dict = {
        "data_path": '../Data_Sets/German_Credit_Data/',
        "filename_train": 'german-train.csv',
        "filename_test": 'german-test.csv',
        "filename_test_unit": "german-test.csv",  
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
        "filename_test": 'adult-test-{}.csv'.format(file_number),
        "filename_test_unit": 'adult-test.csv',
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
        "filename_test": 'compas-test-{}.csv'.format(file_number),
        "filename_test_unit": 'compas-test.csv',
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
            "german" : german_dict
        }
    data_meta_info = data_meta_dictionaries[data_name]


    dataset_test = loader.DataLoader_Tabular(path=data_meta_info["data_path"],
                                                     filename=data_meta_info["filename_test"],
                                                     label=data_meta_info["label"], scale = "standard")
    dataset_test_tr = loader.DataLoader_Tabular(path=data_meta_info["data_path"],
                                                     filename=data_meta_info["filename_test_unit"],
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

    # Train Non-Linear model
    train_loader = DataLoader(dataset_train, batch_size = 32, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size = 32, shuffle=False)
    test_loader_tr = DataLoader(dataset_test_tr, batch_size = 32, shuffle=False)

    def epoch_train(loader, model, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.
        for X,y,ind in loader:
            X,y = X.to(device).float(), y.float().to(device)
            yp = model(X)
            loss = nn.BCEWithLogitsLoss()(yp.squeeze(), y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_err += ((yp > 0).squeeze() != y).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    def train_model(num_epochs, path_to_store_model):
        best_err = 1.0
        model_D = NN3_ModelSP(num_input)
        opt = optim.SGD(model_D.parameters(), lr=1e-2)
        print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
        for i in range(num_epochs):
            train_err, train_loss = epoch_train(train_loader, model_D, opt)
            test_err, test_loss = epoch_train(test_loader_tr, model_D)
            if best_err > test_err:
                print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t") 
                torch.save(model_D, path_to_store_model)
                best_err = test_err
        torch.save(model_D, path_to_store_model)

    num_epochs = 100
    train_model(num_epochs, "./models/{}/{}_nn{}_model_D_l5.pth".format(data_name,data_name, n_layers) )


    

    ## LIME Appx Code. 
    test_loader = DataLoader(dataset_test, batch_size = 1, shuffle=False)
    model_appx_sample = []
    model_D = torch.load("./models/{}/{}_nn{}_model_D_l5.pth".format(data_name,data_name, n_layers), map_location=torch.device('cpu'))

    def similarity_kernel(
        original_input: Tensor,
        perturbed_input: Tensor,
        perturbed_interpretable_input: Tensor,
        **kwargs)->Tensor:
            kernel_width = kwargs["kernel_width"]
            l2_dist = torch.norm(original_input - perturbed_input)
            return torch.exp(- (l2_dist**2)) 
        
    
    def perturb_func(
        original_input: Tensor,
        **kwargs)->Tensor:
        variance = 0.1
        step = 0.05
        while True:
            sample_normal = original_input + torch.normal(0, variance, size=original_input.shape)
            if (found_sample_1[0] == False) and (model_D(sample_normal).squeeze() > 0) :
                found_sample_1[0] = True
                break
            elif (found_sample_1[0] == False) and (model_D(sample_normal).squeeze() < 0):
                variance += step
            elif found_sample_1[0]== True:
                break
        return sample_normal

    def to_interp_transform(curr_sample, original_inp,
                                         **kwargs):
        return curr_sample

    def predict_func(X):
        return (model_D(X) > 0).int() #.unsqueeze(0)
    
    found_sample_1 = [False]
    lime_attr_base = LimeBase(predict_func,  SkLearnLinearModel("linear_model.LogisticRegression", C = 10000, solver = "liblinear"),
                             similarity_func=similarity_kernel,
                             perturb_func=perturb_func,
                             perturb_interpretable_space=False,
                             from_interp_rep_transform=None,
                             to_interp_rep_transform=to_interp_transform)
    #compute lime approximations
    if int(is_lime) == 1:
        for X,y,ind in test_loader:
            X,y = X.to(device).float(), y.long().to(device)
            diff = 100
            w_pre = torch.ones(num_input + 1)* 100
            n_samples = 10000
            if model_D(X) < 0:
                print("[LIME] Processing {} ..".format(ind))
                while diff > 5e-3:
                    found_sample_1 = [False]
                    attr_coefs_base = lime_attr_base.attribute(X.float(), target=0, kernel_width=0.5, n_samples = n_samples)
                    appx_w = lime_attr_base.interpretable_model.linear.weight.data
                    appx_b = lime_attr_base.interpretable_model.linear.bias.data
                    w_comb = torch.cat([appx_w, appx_b.unsqueeze(0)], dim = 1).squeeze()
                    diff = torch.norm(w_comb - w_pre)
                    w_pre = w_comb.clone()
                    n_samples *= 2
                    break
                model_appx_sample.append([ind, X, appx_w, appx_b])     


    # Save it into dictionary
    model_appx_sample_dict = dict()
    def get_g_i_H(model_D):
        opt = optim.SGD(model_D.parameters(), lr=1e-3)
        g_is = []
        train_loader = DataLoader(dataset_train, batch_size = 1, shuffle=True)
        for X,y,ind in train_loader:
            X = X.to(device)
            y = y.to(device)
            yp = model_D(X.view(X.shape[0], -1).to(torch.float32))[:,0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(yp, y.float())
            opt.zero_grad()
            loss.backward()
            g_is.append(torch.cat([torch.clone(model_D.linear.weight.grad).squeeze(), torch.clone(model_D.linear.bias.grad)]))
            opt.zero_grad()
        g_is_tensored = torch.stack(g_is)
        train_loader = DataLoader(dataset_train, batch_size = len(dataset_train), shuffle=False)
        for X,y,ind in train_loader:
            X1 = torch.cat([X, torch.ones((X.size(0), 1))], dim=1)
        w = torch.cat([torch.clone(model_D.linear.weight.data).squeeze(), torch.clone(model_D.linear.bias.data)]).unsqueeze(0).T
        w = w.squeeze()
        def loss(w):
            logits = torch.matmul(X1.float(), w.float())
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(dtype=torch.float32))
        H = hessian(loss, w)
        hess = H.numpy()
        hess = (hess + hess.T) / 2
        hess_pinv = np.linalg.pinv(hess)
        hess_reg_inv = np.linalg.inv(hess + 0.01 * np.eye(hess.shape[0]))
        H_inverse = torch.tensor(hess_reg_inv)
        H_g_prod = torch.matmul(g_is_tensored.float(), H_inverse.float())
        return g_is_tensored, H_g_prod, H_inverse
    if int(is_lime) == 1:
        for sample in model_appx_sample:
            model_appx_sample_dict[sample[0].item()] = []
            model_appx_sample_dict[sample[0].item()].append(convert_sklearn_to_torch_model_v2(sample[2], sample[3], num_input, device))
            model_appx_sample_dict[sample[0].item()].append(get_g_i_H(model_appx_sample_dict[sample[0].item()][0]))
        import pickle as pkl
        pkl.dump(model_appx_sample_dict, open("./models/{}/{}_lime_appx_model_dict_{}_nn_{}_sp.pkl".format(data_name, data_name, n_layers, file_number), "wb"))
    model_appx_sample_dict = pkl.load(open("./models/{}/{}_lime_appx_model_dict_{}_nn_{}_sp.pkl".format(data_name, data_name, n_layers, file_number), "rb"))

    

    # Choose lambda for ROAR
    def get_lambda_val(delta_val, model_appx_sample_dict, lamb_iter = 0.1 , step = 2, val_samples = 5):
        count_samples = val_samples
        while count_samples == val_samples:
            print("[LAMBDASEARCH] lambda :", lamb_iter)
            test_loader_tr = DataLoader(dataset_test_tr, batch_size = 1, shuffle=False)
            count_valid = 0
            count_samples = 0
            for X,y,ind in test_loader_tr:
                if ind.item() not in model_appx_sample_dict.keys():
                    continue
                model_lime = model_appx_sample_dict[ind.item()][0]
                coefficients = model_lime.linear.weight.data
                bias = model_lime.linear.bias.data
                robust_recourse = RobustRecourse(W=torch.clone(coefficients).squeeze().numpy(), W0 = torch.clone(bias).numpy(), feature_costs=None, delta_max = delta_val, norm = norm)
                count_samples += 1
                rfe_rr, delta_W_0_005,_  = robust_recourse.get_recourse(X.T.cpu().numpy(), lamb=lamb_iter)
                print(model_lime(rfe_rr.squeeze().float()))
                if model_lime(rfe_rr.squeeze().float()) > 0:
                    count_valid += 1
                if count_samples == val_samples:
                    break
            print(count_valid, val_samples)
            if count_valid == val_samples:
                lamb_iter *= step
            else:
                break
        return lamb_iter
    
        # A and get k min values 
    def get_A(x, H_g_prod):
        return torch.matmul(H_g_prod.float(), torch.cat([x.float(), torch.ones((1, x.size(1)))], dim=0)).squeeze()


    def get_k_min_values(A, m, k, gs_tensored, H_inverse):
        if m == 0:
            indices =  (A < torch.kthvalue(A, k+1).values)
        else:
            indices =  (A >= torch.kthvalue(A, m).values) and (A < torch.kthvalue(A, m+k+1).values)
        gs_filtered = torch.clone(gs_tensored[indices])
        return torch.sum(torch.matmul(gs_filtered.float(), H_inverse.float()), dim = 0).float()/total_train_samples


    
    # Compute cost and validity of counterfactual generated robust to data removal
    K_percentages = [0, 0.5, 1 , 2, 3 , 5] 
    k_vals = {i : int((i * total_train_samples)/100) for i in K_percentages if i != 0}
    explanation_methods = ["C-CHVAE", "ROAR-0.1", "SCFE"] + ["RR-CFE-{}".format(i) for i in k_vals.keys()] 
    cfe_test_set = dict()
    for explanation_method in explanation_methods:
        cfe_test_set[explanation_method] = []
    
    
    # Compute Explanations
    test_loader_tr = DataLoader(dataset_test_tr, batch_size = len(dataset_test_tr), shuffle=False)

    for X,y,ind in test_loader_tr:
        total_invalid_samples = (model_D(X.float()) < 0).squeeze().sum()

    deltas = [0.1] 
    if int(find_lambda_roar) == 1:
        lamb_iters = dict()
        for delta_val in deltas:
            print("Processing {}...".format(delta_val))
            lamb_iters[delta_val] = get_lambda_val(delta_val, model_appx_sample_dict, val_samples = max([min(int(total_invalid_samples*0.05), 20), 5]))
        pkl.dump(lamb_iters, open("./hyperparameters/{}_hyper_params_delta_{}_ROAR.pkl".format(data_name, norm), "wb"))
    else:
        lamb_iters = pkl.load(open("./hyperparameters/{}_hyper_params_delta_{}_ROAR.pkl".format(data_name, norm), "rb"))

    test_loader = DataLoader(dataset_test, batch_size = 1, shuffle=False)
    model_D = torch.load("./models/{}/{}_nn{}_model_D_l5.pth".format(data_name,data_name, n_layers), map_location=torch.device('cpu'))

    print("Processing norm : ", norm )
    count_samples = 0
    target_delta = 0.00
    for X,y,ind in test_loader:
        print(ind.item())
        if ind.item() in model_appx_sample_dict.keys():
            count_samples += 1
            print("Processing {} sample".format(ind))
            X = X.to(device)
            y = y.to(device)
            yp = model_appx_sample_dict[ind.item()][0](X.float().view(X.shape[0], -1)).max(dim=1)[1].item()
            if yp == 1:
                for k_val in [0.5, 5]:#k_vals.keys():
                    cfe_test_set["RR-CFE-{}".format(k_val)].append([X.float().squeeze().cpu().numpy(), torch.norm(X.float().squeeze().to(device) - X.float().squeeze().to(device), p = norm).cpu().item(), X.squeeze().float()])
                for delta_val in deltas: 
                    cfe_test_set["ROAR-{}".format(delta_val)].append([X.float().squeeze().numpy(), torch.norm(X.float().squeeze() - X.cpu().float().squeeze(), p = norm).item(), X.squeeze().float()])
            else:  
                model_lime = model_appx_sample_dict[ind.item()][0]
                H_g_prod = model_appx_sample_dict[ind.item()][1][1]
                g_is_tensored = model_appx_sample_dict[ind.item()][1][0]
                H_inverse = model_appx_sample_dict[ind.item()][1][2]
                # Compute G_i and H_product 
                for k_val in k_vals.keys():
                    print("processing .. ", k_val)
                    A_m = get_A(X.T.to(device), H_g_prod.to(device))
                    fmk_params = torch.cat([model_lime.linear.weight.data.squeeze(), model_lime.linear.bias.data]) + get_k_min_values(A_m, 0, k_vals[k_val], g_is_tensored, H_inverse)
                    model_appx = LR_Model(num_input).to(device)
                    model_appx.linear.weight.data = fmk_params[:-1].unsqueeze(0)
                    model_appx.linear.bias.data = fmk_params[-1]
                    rr_cfe = SCFE_modified(model_appx, model_lime, _lambda = 2.5, step=2, max_iter = 1000000, lr = 0.5 , target_threshold = target_delta, norm = norm, iter_per_opt = 100 )
                    cfe_rr, cfe_distance_rr = rr_cfe.generate_counterfactuals(X.float())
                    cfe_test_set["RR-CFE-{}".format(k_val)].append([cfe_rr.squeeze().cpu().numpy(), torch.norm(cfe_rr.squeeze().to(device) - X.float().squeeze().to(device), p = norm).cpu().item(), X.squeeze().float()])

                # ROAR
                coefficients = model_lime.linear.weight.data
                bias = model_lime.linear.bias.data
                for delta_val in deltas: 
                    robust_recourse = RobustRecourse(W=torch.clone(coefficients).squeeze().numpy(), W0 = torch.clone(bias).numpy(), feature_costs=None, delta_max = delta_val, norm = norm)
                    roar_cfe, delta_W,_  = robust_recourse.get_recourse(X.T.cpu().numpy(), lamb=lamb_iters[delta_val])
                    cfe_test_set["ROAR-{}".format(delta_val)].append([roar_cfe.squeeze().numpy(), torch.norm(roar_cfe.squeeze() - X.cpu().float().squeeze(), p = norm).item(), X.squeeze().float()])

            # SCFE
            scfe = SCFE_modified(appx_classifier = model_D, _lambda = 2.5, step=4, max_iter = 100000, lr = 0.5 , target_threshold = 0, norm = norm, iter_per_opt = 10 )
            cfe_scfe, cfe_distance_scfe = scfe.generate_counterfactuals(X.float())
            cfe_test_set["SCFE"].append([cfe_scfe.squeeze().cpu().numpy(), torch.norm(cfe_scfe.squeeze().to(device) - X.float().squeeze().to(device), p = norm).cpu().item(), X.squeeze().float()])
            print(cfe_test_set["SCFE"]) 


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
            print(torch.norm(ccvae_cfe.squeeze().to(device) - X.float().squeeze().to(device), p = norm).cpu().item())
            cfe_test_set["C-CHVAE"].append([ccvae_cfe.squeeze().numpy(), torch.norm(ccvae_cfe.squeeze().to(device) - X.float().squeeze().to(device), p = norm).cpu().item(), X.squeeze().float()])
                
            if count_samples%2 == 0:
                pkl.dump(cfe_test_set, open( "./explanations/{}/cost_cfe_dict_explanation_methods_{}_hps_{}_file_name_{}_nn_sp_test_ch.pkl".format(data_name, data_name, "_".join([str(i) for i in K_percentages]), str(file_number)), "wb"))


    pkl.dump(cfe_test_set, open( "./explanations/{}/cost_cfe_dict_explanation_methods_{}_hps_{}_file_name_{}_nn_sp_test_ch.pkl".format(data_name,data_name, "_".join([str(i) for i in K_percentages]), str(file_number)), "wb"))
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.file_number, args.data_name, beta = args.beta,  is_lime = args.is_lime, find_lambda_roar = args.find_lambda_roar)

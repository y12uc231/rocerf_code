
import torch
import numpy as np
from torch import nn
import datetime

import Recourse_Methods.Generative_Model.model as model_vae
from numpy import linalg as LA
import torch.optim as optim
import torch
import numpy as np
from torch import nn
import datetime
from scipy.optimize import linprog
from torch.autograd import Variable
from torch.autograd import grad
import datetime

# Recourse Method - Baseline 1
class RobustRecourse():
    def __init__(self, W=None, W0=None, max_iters= 50000, y_target=[1],
                 delta_max=0.1, feature_costs=None,
                 pW=None, pW0=None, norm = 1):
        self.set_W(W)
        self.set_W0(W0)
        self.norm = norm
        self.set_pW(pW)
        self.set_pW0(pW0)

        self.y_target = torch.tensor(y_target).float()
        self.delta_max = delta_max
        self.feature_costs = feature_costs
        self.max_iters = max_iters
        if self.feature_costs is not None:
            self.feature_costs = torch.from_numpy(feature_costs).float()

    def set_W(self, W):
        self.W = W
        if W is not None:
            self.W = torch.from_numpy(W).float()

    def set_W0(self, W0):
        self.W0 = W0
        if W0 is not None:
            self.W0 = torch.from_numpy(W0).float()

    def set_pW(self, pW):
        self.pW = pW
        if pW is not None:
            self.pW = torch.from_numpy(pW).float()

    def set_pW0(self, pW0):
        self.pW0 = pW0
        if pW0 is not None:
            self.pW0 = torch.from_numpy(pW0).float()

    def l1_cost(self, x_new, x):
        cost = torch.dist(x_new, x, 1)
        return cost
    
    def norm_cost(self, x_new, x):
        cost = torch.dist(x_new, x, self.norm)
        return cost

    def pfc_cost(self, x_new, x):
        cost = torch.norm(self.feature_costs * (x_new - x), 1)
        return cost

    def calc_delta_opt(self, recourse):
        """
		calculate the optimal delta using linear program
		:returns: torch tensor with optimal delta value
		"""
        
        
        W = torch.cat((self.W, self.W0), 0)  # Add intercept to weights
        recourse = torch.cat((recourse, torch.ones(1).unsqueeze(0)), 0)  # Add 1 to the feature vector for intercept

        loss_fn = torch.nn.BCELoss()

        A_eq = np.empty((0, len(W)), float)

        b_eq = np.array([])

        W.requires_grad = True
        f_x_new = torch.nn.Sigmoid()(torch.matmul(W, recourse))
        w_loss = loss_fn(f_x_new, self.y_target)
        gradient_w_loss = grad(w_loss, W)[0]

        c = list(np.array(gradient_w_loss) * np.array([-1] * len(gradient_w_loss)))
        bound = (-self.delta_max, self.delta_max)
        bounds = [bound] * len(gradient_w_loss)

        res = linprog(c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='simplex')
        delta_opt = res.x  # the delta value that maximizes the function
        delta_W, delta_W0 = np.array(delta_opt[:-1]), np.array([delta_opt[-1]])
        return delta_W, delta_W0

    def get_recourse(self, x, lamb=0.1):
        torch.manual_seed(0)

        # returns x'
        x = torch.from_numpy(x).float()
        lamb = torch.tensor(lamb).float()

        x_new = Variable(x.clone(), requires_grad=True)
        optimizer = optim.Adam([x_new])

        loss_fn = torch.nn.BCELoss()

        # Placeholders
        loss = torch.tensor(1)
        loss_diff = 1
        x_diff = 1
        f_x_new = 0

        x_diff_values = []
        n_iters = self.max_iters 
        while loss_diff > 1e-4:
            loss_prev = loss.clone().detach()
            x_prev = x_new.clone().detach()

            delta_W, delta_W0 = self.calc_delta_opt(x_new)
            delta_W, delta_W0 = torch.from_numpy(delta_W).float(), torch.from_numpy(delta_W0).float()

            optimizer.zero_grad()
            if self.pW is not None:
                dec_fn = torch.matmul(self.W + delta_W, x_new) + self.W0
                f_x_new = torch.nn.Sigmoid()(torch.matmul(self.pW, dec_fn.unsqueeze(0)) + self.pW0)[0]
            else:
                f_x_new = torch.nn.Sigmoid()(torch.matmul(self.W + delta_W, x_new) + self.W0 + delta_W0)

            if self.feature_costs is not None:
                cost = self.pfc_cost(x_new, x)
            else:
                cost = self.norm_cost(x_new, x)

            
            loss = loss_fn(f_x_new, self.y_target) + lamb * cost
            loss.backward()
            optimizer.step()

            loss_diff = torch.dist(loss_prev, loss, 2)
            x_diff = torch.dist(x_prev, x_new, 2)
            n_iters -= 1
            x_diff_values.append((torch.linalg.vector_norm(x_prev - x_new) / torch.linalg.vector_norm(x_new)).detach().item())
                
        return x_new.detach(), x_diff_values, np.concatenate((delta_W.detach().numpy(), delta_W0.detach().numpy()))

    # Heuristic for picking hyperparam lambda
    def choose_lambda(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        lambdas = np.arange(0.1, 1.1, 0.1)

        v_old = 0
        for i, lamb in enumerate(lambdas):
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train, x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    r, _ = self.get_recourse(x, lamb)

                    self.set_W(None)
                    self.set_W0(None)
                else:
                    r, _ = self.get_recourse(x, lamb)
                recourses.append(r)

            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            if v >= v_old:
                v_old = v
            else:
                li = max(0, i - 1)
                return lambdas[li]

        return lamb

    def choose_delta(self, recourse_needed_X, predict_fn, X_train=None,
                     predict_proba_fn=None, lamb=0.1):
        deltas = [0.1, 0.25, 0.5, 0.75]

        v_old = 0
        for i, d in enumerate(deltas):
            print("Testing delta:%f" % d)
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train, x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    self.delta_max = d

                    r, _ = self.get_recourse(x, lamb)

                    self.set_W(None)
                    self.set_W0(None)
                else:
                    self.delta_max = d
                    r, _ = self.get_recourse(x, lamb)
                recourses.append(r)

            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            if v >= v_old:
                v_old = v
            else:
                di = max(0, i - 1)
                return deltas[di]

        return d

    def choose_params(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        def get_validity(d, l, recourse_needed_X, predict_proba_fn):
            print("Testing delta %f, lambda %f" % (d, l))
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                self.delta_max = d

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train, x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    r, _ = self.get_recourse(x, l)

                    self.set_W(None)
                    self.set_W0(None)

                else:
                    r, _ = self.get_recourse(x, l)

                recourses.append(r)
            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            return v

        deltas = [0.01, 0.25, 0.5, 0.75]
        lambdas = [0.25, 0.5, 0.75, 1]

        m1_validity = np.zeros((4, 4))
        costs = np.zeros((4, 4))

        delta = None
        lamb = None
        for li, l in enumerate(lambdas):
            if li == 0:
                for di, d in enumerate(deltas):
                    d = deltas[di]
                    v = get_validity(d, l, recourse_needed_X, predict_proba_fn)
                    if v < m1_validity[max(0, di - 1)][li]:
                        di = max(0, di - 1)
                        delta = deltas[di]
                        break
                    m1_validity[di][li] = v

                if delta is None:
                    delta = d
            else:
                v = get_validity(delta, l, recourse_needed_X, predict_proba_fn)
                m1_validity[di][li] = v
                if v < m1_validity[di][max(0, li - 1)]:
                    li = max(0, li - 1)
                    lamb = lambdas[li]
                    break
        if lamb is None:
            lamb = l

        return delta, lamb




        



        
# Baseline 2 : C-CHVAE 
# Second class of counter-factual explanation methods         
class CCHVAE:

    def __init__(self, classifier, model_vae, target_threshold: float = 0,
                 n_search_samples: int = 10, p_norm: int = 2,
                 step: float = 0.05, max_iter: int = 1000, clamp: bool = True):
        
        super().__init__()
        self.classifier = classifier
        self.generative_model = model_vae
        self.n_search_samples = n_search_samples
        self.p_norm = p_norm
        self.step = step
        self.max_iter = max_iter
        self.clamp = clamp
        self.target_treshold = target_threshold
        

    def hyper_sphere_coordindates(self, instance, high, low):
    
        """
        :param n_search_samples: int > 0
        :param instance: numpy input point array
        :param high: float>= 0, h>l; upper bound
        :param low: float>= 0, l<h; lower bound
        :param p: float>= 1; norm
        :return: candidate counterfactuals & distances
        """
    
        delta_instance = np.random.randn(self.n_search_samples, instance.shape[1])
        dist = np.random.rand(self.n_search_samples) * (high - low) + low  # length range [l, h)
        norm_p = LA.norm(delta_instance, ord=self.p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = np.multiply(delta_instance, d_norm)
        candidate_counterfactuals = instance + delta_instance
    
        return candidate_counterfactuals, dist

    def generate_counterfactuals(self, query_instance: torch.tensor, target_class: int = 1) -> torch.tensor:
        """
        :param instance: np array
        :return: best CE
        """  #

        # init step size for growing the sphere
        low = 0
        high = low + self.step

        # counter
        count = 0
        counter_step = 1
        query_instance = query_instance.detach().numpy()

        # get predicted label of instance
        self.classifier.eval()
        instance_label = 1 - target_class
        # vectorize z
        z = self.generative_model.encode_csearch(torch.from_numpy(query_instance).float()).detach().numpy()
        z_rep = np.repeat(z.reshape(1, -1), self.n_search_samples, axis=0)

        while True:
            count = count + counter_step
            if count > self.max_iter:
                candidate_counterfactual_star = np.empty(query_instance.shape[0], )
                candidate_counterfactual_star[:] = np.nan
                distance_star = -1
                print('No CE found')
                break

            # STEP 1 -- SAMPLE POINTS on hypersphere around instance
            latent_neighbourhood, _ = CCHVAE.hyper_sphere_coordindates(self, z_rep, high, low)
            x_ce = self.generative_model.decode_csearch(torch.from_numpy(latent_neighbourhood).float()).detach().numpy()
            

            # STEP 2 -- COMPUTE l1 & l2 norms
            if self.p_norm == 1:
                distances = np.abs((x_ce - query_instance)).sum(axis=1)
            elif self.p_norm == 2:
                distances = LA.norm(x_ce - query_instance, ord=self.p_norm, axis=1)
            else:
                print('Distance not defined yet')
            
            y_logits = torch.stack([torch.tensor([i[0]]) for i in self.classifier(torch.from_numpy(x_ce).float()).detach().numpy()])
            y_candidate = torch.stack([torch.tensor([int(i[0])]) for i in self.classifier(torch.from_numpy(x_ce).float()).detach().numpy() > self.target_treshold])
            

            indeces_total = np.where(y_candidate != instance_label)
            indeces = indeces_total[0]
            if len(indeces) > 0:
                pass

            candidate_counterfactuals = x_ce[indeces]
            candidate_dist = distances[indeces]
            if len(candidate_dist) == 0:  # no candidate found & push search range outside
                low = high
                high = low + self.step
            elif len(candidate_dist) > 0:  # certain candidates generated
                min_index = np.argmin(candidate_dist)
                candidate_counterfactual_star = candidate_counterfactuals[min_index]
                distance_star = candidate_dist[min_index] # np.abs(candidate_counterfactual_star - query_instance).sum()
                break

        return torch.tensor(candidate_counterfactual_star), torch.tensor(distance_star)    
    

 
# Proposed Recourse Method 


class SCFE_modified:
    
    def __init__(self, appx_classifier, lime_classifier = None, device = torch.device("cpu") , target_threshold: float = 0, _lambda: float = 10.0,
                 lr: float = 0.001, max_iter: int = 500, step: float = 2, norm: int = 2, iter_per_opt = 10):
        
        super().__init__()
        self.appx_classification = appx_classifier
        self.lime_classifier = lime_classifier
        self.nn_classifier = model_D
        self.lr = lr
        self.max_iter = max_iter
        self.norm = norm
        self.target_thres = target_threshold
        self._lambda = _lambda
        self.step = step
        self.device = device
        self.iter_per_opt = iter_per_opt
    
    # Returns index of x in arr if present, else -1
    def binary_search(self, low_lambda, high_lambda, cf, query_instance, num_iter):
        # Check base case
        print("Entering binary search on lambda.. ")
        cfes_iter = []
        dist_iter = []
        loss_iter = [] 
        cfe = None
        num_iter_run = num_iter
        
        while (high_lambda > low_lambda) and (num_iter_run <= self.max_iter ):
            self._lambda = (low_lambda + high_lambda) / 2
            cfe, cfe_l , dist_val, loss_val , iters  = self._wachter_optimization(cf, query_instance)
            num_iter_run += iters

            if cfe != None: # case of valid counterfactual
                cfes_iter.extend(cfe_l)
                dist_iter.extend(dist_val)
                loss_iter.extend(loss_val)
                low_lambda = self._lambda
            else:
                high_lambda = self._lambda
        
        return cfe, cfes_iter, dist_iter, loss_iter, num_iter_run
    
    
    def _call_appx_classification(self, cf_candidate):
        output = self.appx_classification(cf_candidate)[0]
        return output
    
    def _call_nn_classification(self, cf_candidate):
        output = self.nn_classifier(cf_candidate)[0]
        return output
    
    def _call_lime_appx_classification(self, cf_candidate):
        output = self.lime_classifier(cf_candidate)[0]
        return output

    def compute_loss(self, _lambda: float, cf_candidate: torch.tensor, original_instance: torch.tensor ) -> torch.tensor:
        output = self._call_appx_classification(cf_candidate)
        loss_classification = (torch.nn.functional.relu(self.target_thres - output ))**2
        # distance loss
        loss_distance = torch.norm((cf_candidate - original_instance), self.norm)
        # full loss
        total_loss = loss_classification + _lambda * loss_distance
        return total_loss, loss_distance
    
    def _wachter_optimization(self, cf, query_instance, target_class = 1, tol = 1e-5):
        counterfactuals_per_lambda = []
        distances_per_lambda = []
        all_loss_per_lambda = []
        cf_prev = torch.clone(cf)
        num_iter_run = 0
        found_cfe = False
        optim = torch.optim.Adam([cf], lr=self.lr)
        cf = cf.to(self.device)
        query_instance = query_instance.to(self.device)
        while num_iter_run < self.max_iter/self.iter_per_opt : 
                cf.requires_grad = True
                total_loss, loss_distance = self.compute_loss(self._lambda, cf, query_instance )
                
                optim.zero_grad()
                total_loss.backward(retain_graph=True)
                optim.step()
                output = self._call_appx_classification(cf)
                output_lime = self._call_lime_appx_classification(cf)
                # store all counterfactuals
                if self._check_cf_valid(output, output_lime, target_class):
                    found_cfe = True
                    counterfactuals_per_lambda.append(torch.clone(cf.detach().cpu()))
                    distances_per_lambda.append(torch.norm((cf - query_instance), self.norm).detach().cpu())
                    all_loss_per_lambda.append(total_loss.detach().cpu()) 
                
                # stop criteria
                if torch.linalg.vector_norm(cf_prev - cf) / torch.linalg.vector_norm(cf) < tol:
                    break
                cf_prev = torch.clone(cf)
                num_iter_run += 1
        if found_cfe == True:
            return torch.clone(cf) , counterfactuals_per_lambda, distances_per_lambda, all_loss_per_lambda, num_iter_run
        else:
            return [None]*4 + [num_iter_run]
    
    def generate_counterfactuals(self, query_instance: torch.tensor, target_class: int = 1) -> torch.tensor:
        """
            query instance: the point to be explained
            target_class: Direction of the desired change. If target_class = 1, we aim to improve the score,
                if target_class = 0, we aim to decrese it (in classification and regression problems).
            _lambda: Lambda parameter (distance regularization) parameter of the problem
        """
        
        cf = query_instance.clone().requires_grad_(True)
        counterfactuals = []
        distances = []
        all_loss = []
        prev_inc = False
        curr_inc = False
        lambda_old = self._lambda
        num_iter = 0
        num_calls = 0
        while num_iter < self.max_iter :
            iterations_per_lambda = 0
            cfe, counterfactuals_iter, distances_iter, all_loss_iter, num_iter_run  = self._wachter_optimization(cf, query_instance)
            num_calls += 1   
            num_iter += num_iter_run
            if cfe != None:
                counterfactuals.extend(counterfactuals_iter)
                distances.extend(distances_iter)
                all_loss.extend(all_loss_iter)
                lambda_old = self._lambda
                self._lambda *= self.step
                prev_inc = curr_inc
                curr_inc = True
            else:
                prev_inc = curr_inc
                curr_inc = False
                if (num_calls >= 2) and (prev_inc == True) and (curr_inc == False):
                    # Binary Search
                    cfe, counterfactuals_iter, distances_iter, all_loss_iter, num_iter_run = self.binary_search(lambda_old, self._lambda, cf, query_instance, num_iter)
                    if cfe != None:
                        counterfactuals.extend(counterfactuals_iter)
                        distances.extend(distances_iter)
                        all_loss.extend(all_loss_iter)
                    break
                lambda_old = self._lambda 
                self._lambda /= self.step
                
        print("Iterations completed : ", num_iter)
        if not len(counterfactuals):
            print('No CE found')
            return None
        
        # Choose the nearest counterfactual
        counterfactuals = torch.stack(counterfactuals)
        distances = torch.stack(distances)
        distances = distances.detach()
        index = torch.argmin(distances)
        counterfactuals = counterfactuals.detach()

        ce_star = counterfactuals[index]
        distance_star = distances[index]
                
        return ce_star, distance_star


    def _check_cf_valid(self, output, output_lime, target_class):
        """ Check if the output constitutes a sufficient CF-example.
            target_class = 1 in general means that we aim to improve the score,
            whereas for target_class = 0 we aim to decrese it.
        """
        if target_class == 1:
            check = (output.item() >= self.target_thres) and (output_lime.item() >= self.target_thres)
            return check
        else:
            check = (output.item() < self.target_thres) and (output_lime.item() < self.target_thres)
            return check
        
    

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)



###################################################################
# Unsupervised Training
###################################################################
def training(model, constraints, optimizer, scheduler, x_tensor, t_tensor, args):
    batch_size = args['batch_size']
    total_iteration = args['total_iteration']
    penalty_coefficient = args['penalty_coefficient']
    distortion_coefficient = args['distortion_coefficient']
    transport_coefficient = args['transport_coefficient']
    volume_list = []
    penalty_list = []
    dist_list = []
    trans_list = []
    bias_tensor = torch.ones(batch_size, x_tensor.shape[1]).to(x_tensor.device) * np.mean(args['bound'])
    model.train()
    for n in range(total_iteration):
        optimizer.zero_grad()
        batch_index = np.random.choice([i for i in range(x_tensor.shape[0])],  batch_size, replace=True)
        x_input = x_tensor[batch_index]
        batch_index = np.random.choice([i for i in range(t_tensor.shape[0])], batch_size, replace=True)
        t_input = t_tensor[batch_index]
        # x_input.requires_grad = True
        n_dim = x_input.shape[1]
        if args['scale_ratio']>1:
            xt, logdet, _ = model(x_input, t_input)
            _, _, logdis = model((x_input-bias_tensor)*args['scale_ratio']+bias_tensor, t_input)
        else:
            xt, logdet, logdis = model(x_input, t_input)
        trans = torch.mean((x_input - xt) ** 2, dim=1, keepdim=True)
        volume = logdet # * unit_volume
        xt_scale = constraints.scale(t_input, xt)
        xt_full = constraints.complete_partial(t_input, xt_scale)
        violation = constraints.cal_penalty(t_input, xt_full)
        penalty = torch.sum(torch.abs(violation), dim=-1, keepdim=True)
        loss = -  torch.mean(volume)  \
                +  penalty_coefficient * torch.mean(penalty) \
                +  distortion_coefficient * torch.mean(logdis) \
                +  transport_coefficient * torch.mean(trans)
        loss.backward()
        optimizer.step()
        scheduler.step()
        volume_list.append(torch.mean(logdet).detach().cpu().numpy()/n_dim)
        penalty_list.append(torch.mean(penalty).detach().cpu().numpy())
        dist_list.append(torch.mean(logdis).detach().cpu().numpy()/args['num_layer'])
        trans_list.append(torch.mean(trans).detach().cpu().numpy())
        if n%1000==0 and n>0:
            model.eval()
            with torch.no_grad():
            # bias_tensor.requires_grad = True
                x0,_,_ = model(bias_tensor, t_input)
                x0_scale = constraints.scale(t_input, x0)
                x0_full = constraints.complete_partial(t_input, x0_scale, backward=False)
                violation_0 = constraints.cal_penalty(t_input, x0_full)
                penalty_0 = torch.sum(torch.abs(violation_0), dim=-1, keepdim=True)
            print(f'Iteration: {n}/{total_iteration}, '
                  f'Volume: {volume_list[-1]:.4f}, '
                  f'Penalty: {penalty_list[-1]:.4f}, '
                  f'Distortion: {dist_list[-1]:.4f}, '
                  f'Transport: {trans_list[-1]:.4f}, '
                  f'Valid: {torch.mean(penalty_0).detach().cpu().numpy():.8f}',
                  end='\n')
    return model, volume_list, penalty_list, dist_list, trans_list




###################################################################
# Binary Search for Homeomorphic Projection
###################################################################
def homeo_bisection(model, constraints, args, x_tensor, t_tensor):
    model.eval()
    steps = args['proj_para']['corrTestMaxSteps']
    eps = args['proj_para']['corrEps']
    bis_step = args['proj_para']['corrBis']
    with torch.no_grad():
        bias = torch.tensor(np.mean(args['mapping_para']['bound']), device=x_tensor.device).view(1, -1)
        t_tensor_infeasible = t_tensor
        x_tensor_infeasible = x_tensor
        x_latent_infeasible, _, _ = model.backward(x_tensor_infeasible, t_tensor_infeasible)
        alpha_upper = torch.ones([t_tensor_infeasible.shape[0], 1], device=x_tensor.device)
        alpha_lower = torch.zeros([t_tensor_infeasible.shape[0], 1], device=x_tensor.device)
        for k in range(steps):
            alpha = (1-bis_step)*alpha_lower + bis_step*alpha_upper
            xt, _,_ = model(alpha * (x_latent_infeasible - bias) + bias, t_tensor_infeasible)
            xt_scale = constraints.scale(t_tensor_infeasible, xt)
            xt_full = constraints.complete_partial(t_tensor_infeasible, xt_scale, backward=False)
            violation = constraints.cal_penalty(t_tensor_infeasible, xt_full)
            penalty = torch.max(torch.abs(violation), dim=1)[0]
            sub_feasible_index = (penalty < eps)
            sub_infeasible_index = (penalty >= eps)
            alpha_lower[sub_feasible_index] = alpha[sub_feasible_index]
            alpha_upper[sub_infeasible_index] = alpha[sub_infeasible_index]
            if (alpha_upper-alpha_lower).max()<1e-2:
                break
        xt, _, _ = model(alpha_lower * (x_latent_infeasible - bias) + bias, t_tensor_infeasible)
        xt_scale = constraints.scale(t_tensor_infeasible, xt)
        xt_full = constraints.complete_partial(t_tensor_infeasible, xt_scale, backward=False)
    return xt_full, k



###################################################################
# Gradient descent 
# Used only at test time, so let PyTorch avoid building the computational graph
###################################################################
def diff_projection(data, X, Y, args):
    take_grad_steps = args['proj_para']['useTestCorr']
    if take_grad_steps:
        lr = args['proj_para']['corrLr']
        eps_converge = args['proj_para']['corrEps']
        max_steps = args['proj_para']['corrTestMaxSteps']
        momentum = args['proj_para']['corrMomentum']
        partial_corr = True if args['proj_para']['corrMode'] == 'partial' else False
        Y_new = Y
        i = 0
        old_step = 0
        while i < max_steps:
            with torch.enable_grad():
                violation = data.cal_penalty(X, Y_new)
                if (torch.max(torch.abs(violation), dim=1)[0].max() < eps_converge):
                    break
                if partial_corr:
                    Y_step = data.ineq_partial_grad(X, Y_new)
                else:
                    ineq_step = data.ineq_grad(X, Y_new)
                    eq_step = data.eq_grad(X, Y_new)
                    Y_step = 0.5 * ineq_step + 0.5 * eq_step
                new_step = lr * Y_step + momentum * old_step
                Y_new = Y_new - new_step
                Y_new = data.complete_partial(X, Y_new[:,data.partial_unknown_vars])
                old_step = new_step
                i += 1
        return Y_new, i
    else:
        return Y, 0






#################################################################
# Constraint for 2-dim toy example
#################################################################


class Convex_Constraints:
    def __init__(self):
        """
        Ax - b <= 0
        """
        self.A = torch.tensor(np.array([[1, -1], [-1, -1], [1, 1], [-1, 1]])).permute(1, 0).to(device=device)
        self.b = torch.tensor(np.array([2, 2, 2, 2])).view(1, 4).to(device=device)

        self.t_test = [[0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 1.7, 1.7], [1.7, 1.7, 0.3, 1.7], [1.7, 1.7, 1.7, 1.7]]
        self.sampling_range = [0, 2]
        self.t_dim = 4

    def forward(self, x):
        violation = torch.matmul(x, self.A) - (self.b)

        return violation

    def cal_penalty(self, t, x):
        violation = torch.matmul(x, self.A) - t

        return violation

    def scale(self, t, x):
        return x

    def complete_partial(self, t, x, backward=True):
        return x

    def plot_boundary(self, t):
        x = np.linspace(-2, 2, 100)
        boundary, _, _, _ = plt.plot(x, x - t[0], x, -x - t[1], x, -x + t[2], x, x + t[3],
                                     linewidth=2, alpha=0.9, c='cornflowerblue', )
        return boundary


class Non_Convex_Constraints:
    def __init__(self):
        """
        A[x,y] - b + t <= 0
        """
        self.A = torch.tensor(np.array([[1, -1], [-1, -1], [1, 1], [-1, 1]])).permute(1, 0).to(device=device)
        self.bias = 2
        self.b = torch.tensor(np.array([self.bias] * 4)).view(1, 4).to(device=device)

        self.t_test = [[1.5, 1.5], [1.5, 0.2], [0.2, 1.5], [0.2, 0.2]]
        self.sampling_range = [0, 2]
        self.t_dim = 2

    def cal_penalty(self, t, x):
        """
        y <= 0.5*x^2 + [0,2]
        y >= -0.5x^2 - [0,2]
        """
        violation = torch.matmul(x, self.A) - (self.b)
        violation_1 = (-0.5 * (x[:, [0]]) ** 2 + x[:, [1]] - t[:, [0]])
        violation_2 = (-0.5 * (x[:, [0]]) ** 2 - x[:, [1]] - t[:, [1]])
        violation = torch.cat([violation, violation_1, violation_2], dim=-1)
        return violation

    def scale(self, t, x):
        return x

    def complete_partial(self, t, x, backward=True):
        return x

    def plot_boundary(self, t):
        tt = self.bias
        x = np.linspace(-2, 2, 100)
        boundary, _, _ = plt.plot([0, tt, 0, -tt, 0], [tt, 0, -tt, 0, tt],
                                  x, 0.5 * x ** 2 + t[0],
                                  x, -0.5 * x ** 2 - t[1], linewidth=2, alpha=0.9, c='cornflowerblue', )
        return boundary


class Complex_Constraints:
    def __init__(self):
        """
        xTx <= 0
        """
        self.A = torch.tensor(np.array([[1., -1.], [-1., -1.], [1., 1.], [-1., 1.]])).permute(1, 0).to(device=device)
        # self.bias = 2
        # self.b = torch.tensor(np.array([self.bias] * 4)).view(1, 4).to(device=device)
        # self.t_test = [ [1.5, 1.5, 0.5, 1.5, 1.5, 0.5],
        #                 [1.5, 0.5, 1.5, 1.5, 0.5, 0.5],
        #                 [0.5, 1.5, 0.5, 0.5, 1.5, 0.5],
        #                 [0.5, 0.5, 1.5, 1.5, 1.5, 0.5]]
        self.sampling_range = [0.1, 2]
        self.t_dim = 6 + 2
        self.n_case = 8
        self.t_test = np.random.uniform(low=self.sampling_range[0],
                                        high=self.sampling_range[1],
                                        size=[self.n_case,self.t_dim])

    def cal_penalty(self, t, x):
        """
        y <= 0.5*x^2 + [0,2]
        y >= -0.5x^2 - [0,2]
        """
        bias = t[:,2:6]
        violation = torch.matmul(x, self.A) - (bias)
        violation_1 = (-t[:,[6]] * (x[:, [0]]) ** 2 + x[:, [1]] - t[:, [0]])
        violation_2 = (-t[:,[7]] * (x[:, [0]]) ** 2 - x[:, [1]] - t[:, [1]])
        violation = torch.cat([violation, violation_1, violation_2], dim=-1)
        return torch.clamp(violation, 0)

    def scale(self, t, x):
        return x

    def complete_partial(self, t, x, backward=True):
        return x

    def plot_boundary(self, t):
        x = np.linspace(-5, 5, 1000)
        plt.plot(x,  x - t[2], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x, -x - t[3], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x, -x + t[4], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x,  x + t[5], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x, t[6] * x ** 2 + t[0], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x, -t[7] * x ** 2 - t[1], linewidth=1.5, alpha=0.7, c='cornflowerblue')


class Box_Constraints:
    def __init__(self):
        """
        -t1 < x < t1
        -t2 < y < t2
        """
        self.t_test = [[0.2, 1.8], [0.8, 1.2], [1.2, 0.8], [1.8, 0.2]]
        self.sampling_range = [0, 2]
        self.t_dim = 2

    def cal_penalty(self, t, x):
        violation = [torch.relu(x[:, [0]] - t[:, [0]]),
                     torch.relu(x[:, [1]] - t[:, [1]]),
                     torch.relu(-x[:, [0]] - t[:, [0]]),
                     torch.relu(-x[:, [1]] - t[:, [1]])]
        violation = torch.cat(violation, dim=-1)
        return violation

    def scale(self, t, x):
        return x

    def complete_partial(self, t, x, backward=True):
        return x

    def plot_boundary(self, t):
        t1 = t[0]
        t2 = t[1]
        boundary = plt.plot([t1, t1, -t1, -t1, t1], [t2, -t2, -t2, t2, t2], linewidth=2, alpha=0.9, c='cornflowerblue')
        return boundary




###################################################################
# NN predictor
###################################################################
import torch.nn as nn
import operator
from functools import reduce

class NNSolver(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, args):
        super().__init__()
        self._args = args
        layer_sizes = [in_dim]+ [hidden_dim] * self._args['nn_para']['num_layer']
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], out_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value


import pandas as pd
import os
def csv_record(epoch_stats, data, args):
    record_file = 'results/all_record.csv'
    labels = ['Prob', 'Algo',
              'Fea_rate',
              'Ineq_vio', 'Ineq_vio_rate',
              'Eq_vio', 'Eq_vio_rate',
              'Sol_MAE', 'Sol_MAPE', 'Infea_Sol_MAE', 'Infea_Sol_MAPE',
              'Obj_MAE', 'Obj_MAPE', 'Infea_Obj_MAE', 'Infea_Obj_MAPE',
              'Ave_time', 'Ave_speedup',
              'Ave_porj_time', 'Ave_proj_sppedup',
              'Ave_raw_time', 'Ave_raw_speedup']
    if not os.path.exists(record_file):
        data_record = pd.DataFrame(columns=labels)
        data_record.loc[0] = [str(0)]*len(labels)
    else:
        data_record = pd.read_csv(record_file, index_col=False)
    ### Record pure NN prediction & x-Proj post-processing
    infeasible_index = epoch_stats['valid_index_infeasible']

    row_index = (data_record['Prob'] == str(data)) & (data_record['Algo'] == 'NN')
    if not row_index.any():
        data_record.loc[data_record.shape[0]] = {'Prob': str(data), 'Algo': 'NN'}
        row_index = (data_record['Prob'] == str(data)) & (data_record['Algo'] == 'NN')
    data_record.loc[row_index, 'Fea_rate'] = round((1-np.mean(epoch_stats['valid_raw_vio_instance']))*100, 2)
    data_record.loc[row_index, 'Ineq_vio'] = round(np.mean(epoch_stats['valid_raw_ineq_sum'][infeasible_index]), 3)
    data_record.loc[row_index, 'Ineq_vio_rate'] = round(np.mean(epoch_stats['valid_raw_ineq_num_viol_0'][infeasible_index])/data.nineq*100,2)
    data_record.loc[row_index, 'Eq_vio'] = round(np.mean(epoch_stats['valid_raw_eq_sum'][infeasible_index]), 3)
    data_record.loc[row_index, 'Eq_vio_rate'] = round(np.mean(epoch_stats['valid_raw_eq_num_viol_0'][infeasible_index])/data.neq*100, 2)

    data_record.loc[row_index, 'Sol_MAE'] = round(np.mean(epoch_stats['valid_raw_mae_loss']), 2)
    data_record.loc[row_index, 'Sol_MAPE'] = round(np.mean(epoch_stats['valid_raw_mape_loss'])*100, 2)
    data_record.loc[row_index, 'Infea_Sol_MAE'] = round(np.mean(epoch_stats['valid_raw_mae_loss'][infeasible_index]), 2)
    data_record.loc[row_index, 'Infea_Sol_MAPE'] = round(np.mean(epoch_stats['valid_raw_mape_loss'][infeasible_index])*100, 2)

    data_record.loc[row_index, 'Obj_MAE'] = round(np.mean(epoch_stats['valid_raw_obj_mae']), 2)
    data_record.loc[row_index, 'Obj_MAPE'] = round(np.mean(epoch_stats['valid_raw_obj_mape'])*100, 2)
    data_record.loc[row_index, 'Infea_Obj_MAE'] = round(np.mean(epoch_stats['valid_raw_obj_mae'][infeasible_index]), 2)
    data_record.loc[row_index, 'Infea_Obj_MAPE'] = round(np.mean(epoch_stats['valid_raw_obj_mape'][infeasible_index])*100, 2)

    data_record.loc[row_index, 'Ave_time'] =  round(epoch_stats['batch_solve_raw_time'], 4)
    data_record.loc[row_index, 'Ave_speedup'] = round(epoch_stats['batch_raw_speed_up'], 1)
    # data_record.loc[row_index, 'Ave_raw_time'] = round(epoch_stats['batch_solve_raw_time'], 4)
    # data_record.loc[row_index, 'Ave_raw_speedup'] = round(epoch_stats['batch_raw_speed_up'], 1)
    # data_record.loc[row_index, 'Ave_porj_time'] = round(epoch_stats['batch_solve_proj_time'], 4)
    # data_record.loc[row_index, 'Ave_proj_sppedup'] = round(epoch_stats['batch_proj_speed_up'], 1)




    row_index = (data_record['Prob'] == str(data)) & (data_record['Algo'] == args['projType'])
    if not row_index.any():
        data_record.loc[data_record.shape[0]] = {'Prob': str(data), 'Algo': args['projType']}
        row_index = (data_record['Prob'] == str(data)) & (data_record['Algo'] == args['projType'])
    data_record.loc[row_index, 'Fea_rate'] = round((1-np.mean(epoch_stats['valid_cor_vio_instance']))*100, 2)
    data_record.loc[row_index, 'Ineq_vio'] = round(np.mean(epoch_stats['valid_cor_ineq_sum'][infeasible_index]), 3)
    data_record.loc[row_index, 'Ineq_vio_rate'] = round(np.mean(epoch_stats['valid_cor_ineq_num_viol_0'][infeasible_index])/data.nineq*100,2)
    data_record.loc[row_index, 'Eq_vio'] = round(np.mean(epoch_stats['valid_cor_eq_sum'][infeasible_index]), 3)
    data_record.loc[row_index, 'Eq_vio_rate'] = round(np.mean(epoch_stats['valid_cor_eq_num_viol_0'][infeasible_index])/data.neq*100, 2)

    data_record.loc[row_index, 'Sol_MAE'] = round(np.mean(epoch_stats['valid_cor_mae_loss']), 2)
    data_record.loc[row_index, 'Sol_MAPE'] = round(np.mean(epoch_stats['valid_cor_mape_loss'])*100, 2)
    data_record.loc[row_index, 'Infea_Sol_MAE'] = round(np.mean(epoch_stats['valid_cor_mae_loss'][infeasible_index]), 2)
    data_record.loc[row_index, 'Infea_Sol_MAPE'] = round(np.mean(epoch_stats['valid_cor_mape_loss'][infeasible_index])*100, 2)

    data_record.loc[row_index, 'Obj_MAE'] = round(np.mean(epoch_stats['valid_cor_obj_mae']), 2)
    data_record.loc[row_index, 'Obj_MAPE'] = round(np.mean(epoch_stats['valid_cor_obj_mape'])*100, 2)
    data_record.loc[row_index, 'Infea_Obj_MAE'] = round(np.mean(epoch_stats['valid_cor_obj_mae'][infeasible_index]), 2)
    data_record.loc[row_index, 'Infea_Obj_MAPE'] = round(np.mean(epoch_stats['valid_cor_obj_mape'][infeasible_index])*100, 2)


    data_record.loc[row_index, 'Ave_time'] = round(epoch_stats['batch_solve_time'], 4)
    data_record.loc[row_index, 'Ave_speedup'] = round(epoch_stats['batch_speed_up'], 1)
    # data_record.loc[row_index, 'Ave_raw_time'] = round(epoch_stats['batch_solve_raw_time'], 4)
    # data_record.loc[row_index, 'Ave_raw_speedup'] = round(epoch_stats['batch_raw_speed_up'], 1)
    data_record.loc[row_index, 'Ave_porj_time'] = round(epoch_stats['batch_solve_proj_time'], 4)
    data_record.loc[row_index, 'Ave_proj_sppedup'] = round(epoch_stats['batch_proj_speed_up'], 1)


    data_record.to_csv(record_file, index=False)



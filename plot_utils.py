import numpy as np
import torch
from training_utils import homeo_bisection
from matplotlib import pyplot as plt

torch.set_default_dtype(torch.float64)

###################################################################
# Plot figures
###################################################################

def scatter_constraint_approximation(model, constraints, x_tensor, simple_set, instance_file, paras):
    t_test = constraints.t_test
    t_num = len(t_test)
    model.eval()
    fig = plt.figure(figsize=[t_num*(4+1), 4])
    fig.tight_layout()
    n_samples = x_tensor.shape[0]
    grid = plt.GridSpec(1, t_num+1)
    x = x_tensor.detach().cpu().numpy()
    char_size = 20
    if simple_set=='sphere':
        x_norm = np.linalg.norm(x,ord=2, axis=1).T
        norm_tile = r'$\mathcal{B}$: $2$-norm ball'
    else:
        x_norm = np.linalg.norm(x,ord=np.inf, axis=1).T
        norm_tile = r'$\mathcal{B}$: $\infty$-norm ball'
    plt.subplot(grid[0,0])
    plt.scatter(x[:, 0], x[:, 1], s=0.1, alpha=0.7, c=x_norm, label='Sphere sampling')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xticks([-1,0,1], fontsize=16)
    plt.yticks([-1,0,1], fontsize=16)
    plt.title(norm_tile, fontsize=char_size)
    titles = [r'$\mathcal{K}_{\theta_1}=\Phi(\mathcal{B}|\theta_1)$',
                r'$\mathcal{K}_{\theta_2}=\Phi(\mathcal{B}|\theta_2)$',
                r'$\mathcal{K}_{\theta_3}=\Phi(\mathcal{B}|\theta_3)$',
                r'$\mathcal{K}_{\theta_4}=\Phi(\mathcal{B}|\theta_4)$',
                r'$\mathcal{K}_{\theta_5}=\Phi(\mathcal{B}|\theta_5)$',
                r'$\mathcal{K}_{\theta_6}=\Phi(\mathcal{B}|\theta_6)$',
                r'$\mathcal{K}_{\theta_7}=\Phi(\mathcal{B}|\theta_7)$',
                r'$\mathcal{K}_{\theta_8}=\Phi(\mathcal{B}|\theta_8)$',
                r'$\mathcal{K}_{\theta_9}=\Phi(\mathcal{B}|\theta_9)$',]
    for i, t in enumerate(t_test):
        plt.subplot(grid[0,i+1])
        t_tensor = torch.tensor(t).to(device=x_tensor.device).view(1, -1).repeat(n_samples, 1)
        # x_tensor.requires_grad = True
        xt, _, _ = model(x_tensor, t_tensor)
        x = x_tensor.detach().cpu().numpy()
        xt = xt.detach().cpu().numpy()
        constraints.plot_boundary(t)
        plt.scatter(xt[:, 0], xt[:, 1], s=0.1, alpha=0.7, c=x_norm ,label='Constraint approximation')
        plt.title(titles[i], fontsize=char_size)
        plt.xlim([-2.2, 2.2])
        plt.ylim([-2.2, 2.2])
        plt.xticks([-2,0,2], fontsize=16)
        plt.yticks([-2,0,2], fontsize=16)
    plt.subplots_adjust(wspace=0.15)
    seed = paras['seed']
    shape = paras['init_shape']
    dis_coff = paras['distortion_coefficient']
    plt.savefig(instance_file+f'/{constraints.__class__.__name__}_{shape}_{dis_coff}_{seed}_{len(t_test)}_constraint_approximation_scatter.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_convergence(volume_list, penalty_list, dist_list, trans_list, instance_file):
    index = [i for i in range(len(volume_list))]
    t_num = 3
    fig = plt.figure(figsize=[t_num*(4+1.25),4])
    fig.tight_layout()
    char_size = 20

    plt.subplot(1, t_num, 1)
    plt.plot(index, volume_list, alpha=0.7, c='royalblue', label='Log-det')
    plt.xlabel('Iteration', fontsize=char_size)
    # plt.ylabel('Log-volume', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Log-volume', fontsize=char_size)

    plt.subplot(1, t_num, 2)
    plt.plot(index, penalty_list, alpha=0.7, c='darkorange', label='Penalty')
    plt.xlabel('Iteration', fontsize=char_size)
    # plt.ylabel('Penalty', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Constraint violation', fontsize=char_size)

    plt.subplot(1, t_num, 3)
    plt.plot(index, dist_list, alpha=0.7, c='darkred', label='Dist')
    plt.xlabel('Iteration', fontsize=char_size)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.ylabel('Log-distortion', fontsize=18)
    plt.title('Log-distortion', fontsize=char_size)

    # plt.subplot(1, 4, 4)
    # plt.plot(index, trans_list, alpha=0.7, c='seagreen', label='Dist')
    # plt.xlabel('Iteration', fontsize=15)
    # plt.ylabel('trans_list', fontsize=15)
    # plt.legend(['Transport cost'], fontsize=15)

    # plt.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.subplots_adjust(wspace=0.15)
    plt.savefig(instance_file+'/convergence.png', bbox_inches='tight',  dpi=300)
    # plt.show()
    # plt.close()


def scatter_constraint_evolution(model, constraints, x_tensor, simple_set, instance_file, paras):
    t_test = constraints.t_test
    t_num = paras['num_layer'] + 1
    model.eval()
    fig = plt.figure(figsize=[t_num*(4+1.2),4])
    fig.tight_layout()
    n_samples = x_tensor.shape[0]
    grid = plt.GridSpec(1, t_num+1)
    x = x_tensor.detach().cpu().numpy()
    char_size = 20
    if simple_set=='sphere':
        x_norm = np.linalg.norm(x,ord=2, axis=1).T
        norm_tile = r'$\mathcal{B}$: $2$-norm ball'
    else:
        x_norm = np.linalg.norm(x,ord=np.inf, axis=1).T
        norm_tile = r'$\mathcal{B}$: $\infty$-norm ball'
    plt.subplot(grid[0,0])
    plt.scatter(x[:, 0], x[:, 1], s=0.1, alpha=0.7, c=x_norm, label='Sphere sampling')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xticks([-1,0,1], fontsize=16)
    plt.yticks([-1,0,1], fontsize=16)
    plt.title(norm_tile, fontsize=char_size)
    t = t_test[2]
    t_tensor = torch.tensor(t).to(device=x_tensor.device).view(1, -1).repeat(n_samples, 1)
    xt_list = model.forward_traj(x_tensor, t_tensor)
    k = 1
    for i, xt in enumerate(xt_list):
        if (i+1)%4==0 or i == len(xt_list) - 1 :
            plt.subplot(grid[0, k])
            constraints.plot_boundary(t)
            xt = xt.detach().cpu().numpy()
            plt.scatter(xt[:, 0], xt[:, 1], s=0.1, alpha=0.7, c=x_norm ,label='Constraint evolution')
            plt.xlim([-3.2, 3.2])
            plt.ylim([-3.2, 3.2])
            plt.title(f'block:{k}', fontsize=char_size)
            plt.xticks([-2,0,2], fontsize=16)
            plt.yticks([-2,0,2], fontsize=16)
            k += 1
    plt.subplots_adjust(wspace=0.15)
    seed = paras['seed']
    shape = paras['init_shape']
    plt.savefig(instance_file+f'/{constraints.__class__.__name__}_{shape}_{seed}_{len(t_test)}_constraint_evolution_scatter.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def scatter_projection_error(model, constraints, x_tensor, t_tensor, instance_file, args):
    model.eval()
    proj_error_list =  []
    proj_x_list = []
    h_proj_x_list = []
    h_proj_error_list = []
    parameter_list = []
    penalty_list = []
    char_size = 20
    for i in range(t_tensor.shape[0]):
        t_sub_tensor = (t_tensor[i:i+1]).repeat([x_tensor.shape[0],1])
        x_scale = constraints.scale(t_sub_tensor, x_tensor)
        x_full = constraints.complete_partial(t_sub_tensor, x_scale, backward=False)
        violation = constraints.cal_penalty(t_sub_tensor, x_full)
        penalty = torch.sum(torch.abs(violation), dim=-1, keepdim=True)
        infeasible_index = (penalty > 1e-5).view(-1) & (~torch.isnan(penalty)).view(-1) & (~torch.isinf(penalty)).view(-1)
        x_full = x_full[infeasible_index]
        t_sub_tensor = t_sub_tensor[infeasible_index]
        penalty = penalty[infeasible_index]
        proj_x = constraints.opt_proj(t_sub_tensor, x_full).to(x_full.device).view(x_full.shape)
        h_proj_x, h_steps = homeo_bisection(model, None, constraints, args, x_tensor[infeasible_index], x_full, t_sub_tensor)
        proj_error_list.append(torch.abs(proj_x - x_full).sum(-1))
        proj_x_list.append(proj_x)
        h_proj_x_list.append(h_proj_x)
        h_proj_error_list.append(torch.abs(h_proj_x - x_full).sum(-1))
        penalty_list.append(penalty)
        parameter_list.append(torch.ones(size=[t_sub_tensor.shape[0],1])*i)

    proj_error_list = torch.cat(proj_error_list, dim=0).view(-1).cpu().numpy()
    proj_x_list = torch.cat(proj_x_list, dim=0).cpu().numpy()
    h_proj_x_list = torch.cat(h_proj_x_list, dim=0).cpu().numpy()
    h_proj_error_list = torch.cat(h_proj_error_list, dim=0).view(-1).cpu().numpy()
    penalty_list = torch.cat(penalty_list, dim=0).cpu().view(-1).numpy()
    parameter_list = torch.cat(parameter_list, dim=0).view(-1).cpu().numpy()

    # fig = plt.figure(figsize=[3*(4+1),4])
    # fig.tight_layout()
    # grid = plt.GridSpec(1, 3)
    # plt.subplot(grid[0,0])
    # for i in range(t_tensor.shape[0]):
    #     index = (parameter_list == i) & (~np.isnan(h_proj_error_list))
    #     plt.scatter(penalty_list[index], proj_error_list[index], s=25, alpha=0.5)
    # plt.legend([r'$\theta_1$',r'$\theta_2$',r'$\theta_3$'], loc='upper left', fontsize=char_size-2)
    # plt.xlabel(r'Constraint violation', fontsize=char_size) #: $|\rm{ReLU}(g(x,\theta))|_1$
    # plt.ylabel(r'Proj distance', fontsize=char_size) # : $|x-\rm{Proj}(x)|_1$
    # plt.title(r'Projection', fontsize=char_size)
    # plt.xlim([0, max(penalty_list[~np.isnan(h_proj_error_list)])])
    # plt.ylim([0, max(h_proj_error_list[~np.isnan(h_proj_error_list)])])
    # plt.subplot(grid[0,1])
    # for i in range(t_tensor.shape[0]):
    #     index = (parameter_list == i)& (~np.isnan(h_proj_error_list))
    #     plt.scatter(penalty_list[index], h_proj_error_list[index], s=25, alpha=0.5)
    # plt.legend([r'$\theta_1$',r'$\theta_2$',r'$\theta_3$'], loc='upper left', fontsize=char_size-2)
    # plt.xlabel(r'Constraint violation', fontsize=char_size) # : $|\rm{ReLU}(g(x,\theta))|_1
    # plt.ylabel(r'H-proj distance', fontsize=char_size) # : $|x-\rm{HB}(x)|_1
    # plt.title(r'Homeomorphic projection', fontsize=char_size)
    # plt.xlim([0, max(penalty_list[~np.isnan(h_proj_error_list)])])
    # plt.ylim([0, max(h_proj_error_list[~np.isnan(h_proj_error_list)])])
    # plt.subplot(grid[0,2])

    fig = plt.figure(figsize=[5,4])
    fig.tight_layout()
    for i in range(t_tensor.shape[0]):
        index = (parameter_list == i) & (~np.isnan(h_proj_error_list))
        plt.scatter(proj_error_list[index], h_proj_error_list[index], s=30, alpha=0.4)
    index = ~np.isnan(h_proj_error_list)
    plt.legend([r'$\theta_1$',r'$\theta_2$',r'$\theta_3$'], loc='upper left', fontsize=char_size-4)
    plt.plot([0,max(h_proj_error_list[index])], [0,max(h_proj_error_list[index])], c='powderblue', alpha=0.9, linewidth=3)
    slope = np.mean((h_proj_error_list[index]))/np.mean(proj_error_list[index])
    plt.text(((np.max(h_proj_error_list[index])+np.min(h_proj_error_list[index])))/2, 0,
             r'$\frac{\rm{H-proj\;\;distance}}{\rm{Proj\;\;distance}}\approx$'+'{:.4}'.format(slope),
             horizontalalignment='center',
             fontsize=char_size-4,
             bbox=dict(facecolor='lavender', alpha=0.2))
    # plt.legend(f'slope: {np.mean((h_proj_error_list[index]))/np.mean(proj_error_list[index])}')
    plt.xlabel(r'Proj distance', fontsize=16) # : $|x-\rm{Proj}(x)|_1
    plt.ylabel(r'H-proj distance', fontsize=16) # : $|x-\rm{HfB}(x)|_1
    plt.title(r'Proj vs H-Proj', fontsize=char_size)
    # print('\nH-Proj gap compared with Proj:', np.mean((h_proj_error_list[index])/proj_error_list[index]))
    # print('\nSlope for curve:', np.mean((h_proj_error_list[index]))/np.mean(proj_error_list[index]))
    shape = args['mapping_para']['shape']
    plt.savefig(instance_file+f'/{constraints.__class__.__name__}_{shape}_proj_error_scatter_{x_tensor.shape[1]}.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    return


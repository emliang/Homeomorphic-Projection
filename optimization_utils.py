import torch
from torch.autograd import Function
import numpy as np
import cvxpy as cp
# import ipopt
import copy
import time

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)

###################################################################
# Base PROBLEM
###################################################################
class BaseProblem:
    def __init__(self, dataset, test_size):
        self.input_L = torch.tensor(dataset['XL'] )
        self.input_U = torch.tensor(dataset['XU'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.num = dataset['X'].shape[0]
        self.device = DEVICE
        # self.valid_frac = valid_frac
        # self.test_frac = test_frac

    def eq_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            x = X[n].view(1, -1)
            y = Y[n].view(1, -1)
            y = torch.autograd.Variable(y, requires_grad=True)
            eq_penalty = self.eq_resid(x, y) ** 2
            eq_penalty = torch.sum(eq_penalty, dim=-1, keepdim=True)
            grad = torch.autograd.grad(eq_penalty, y)[0]
            grad_list.append(grad.view(1, -1))
        grad = torch.cat(grad_list, dim=0)
        return grad

    def ineq_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            x = X[n].view(1, -1)
            y = Y[n].view(1, -1)
            y = torch.autograd.Variable(y, requires_grad=True)
            ineq_penalty = self.ineq_resid(x, y) ** 2
            ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
            grad = torch.autograd.grad(ineq_penalty, y)[0]
            grad_list.append(grad.view(1, -1))
        grad = torch.cat(grad_list, dim=0)
        return grad

    def ineq_partial_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            Y_pred = Y[n, self.partial_vars].view(1, -1)
            x = X[n].view(1, -1)
            Y_pred = torch.autograd.Variable(Y_pred, requires_grad=True)
            y = self.complete_partial(x, Y_pred)
            # Y_comp = (x - Y_pred @ self.A_partial.T) @ self.A_other_inv.T
            # y = torch.zeros(1, self.ydim, device=X.device)
            # y[0, self.partial_vars] = Y_pred
            # y[0, self.other_vars] = Y_comp
            ineq_penalty = self.ineq_resid(x, y) ** 2
            ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
            grad_pred = torch.autograd.grad(ineq_penalty, Y_pred)[0]
            grad = torch.zeros(1, self.ydim, device=X.device)
            grad[0, self.partial_vars] = grad_pred
            grad[0, self.other_vars] = - (grad_pred @ self.A_partial.T) @ self.A_other_inv.T
            grad_list.append(grad)
        return torch.cat(grad_list, dim=0)

    def scale_full(self, X, Y):
        # lower_bound = self.L.view(1, -1)
        # upper_bound = self.U.view(1, -1)
        # The last layer of NN is sigmoid, scale to Opt bound
        scale_Y = Y * (self.U - self.L) + self.YL
        return scale_Y

    def scale_partial(self, X, Y):
        # lower_bound = (self.L[self.partial_vars]).view(1, -1)
        # upper_bound = (self.U[self.partial_vars]).view(1, -1)
        scale_Y = Y * (self.U - self.L) + self.L
        return scale_Y

    def scale(self, X, Y):
        if Y.shape[1] < self.ydim:
            Y_scale = self.scale_partial(X, Y)
        else:
            Y_scale = self.scale_full(X, Y)
        return Y_scale

    def cal_penalty(self, X, Y):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y):
        return self.cal_penalty(X, Y)


###################################################################
# QP PROBLEM
###################################################################
class QPProblem(BaseProblem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
                   L<= x <=U
    """
    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.Q_np = dataset['Q']
        self.p_np = dataset['p']
        self.A_np = dataset['A']
        self.G_np = dataset['G']
        self.h_np = dataset['h']
        self.L_np = dataset['YL']
        self.U_np = dataset['YU']
        self.Q = torch.tensor(dataset['Q'] )
        self.p = torch.tensor(dataset['p'] )
        self.A = torch.tensor(dataset['A'] )
        self.G = torch.tensor(dataset['G'] )
        self.h = torch.tensor(dataset['h'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.xdim = dataset['X'].shape[1]
        self.ydim = dataset['Q'].shape[0]
        self.neq = dataset['A'].shape[0]
        self.nineq = dataset['G'].shape[0]
        self.nknowns = 0

        best_partial = dataset['best_partial']
        self.partial_vars = best_partial
        self.partial_unknown_vars = best_partial
        self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars)
        self.A_partial = self.A[:, self.partial_vars]
        self.A_other_inv = torch.inverse(self.A[:, self.other_vars])

        self.trainX = self.X[:-test_size]
        self.testX = self.X[-test_size:]
        self.trainY = self.Y[:-test_size]
        self.testY = self.Y[-test_size:]

    def __str__(self):
        return 'QPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * Y).sum(dim=1)

    def eq_resid(self, X, Y):
        return Y @ self.A.T - X

    def ineq_resid(self, X, Y):
        res = Y @ self.G.T - self.h.view(1, -1)
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def complete_partial(self, X, Y, backward=True):
        Y_full = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y_full[:, self.partial_vars] = Y
        Y_full[:, self.other_vars] = (X - Y @ self.A_partial.T) @ self.A_other_inv.T
        return Y_full

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi in X_np:
                y = cp.Variable(self.ydim)
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  [G @ y <= h, y <= U, y >= L,
                                   A @ y == Xi])
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return sols, total_time, parallel_time

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):

        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                                  [G @ y <= h, y <= U, y >= L,
                                   A @ y == Xi])
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                y.value = y_pred
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  [G @ y <= h, y <= U, y >= L,
                                   A @ y == Xi])
                start_time = time.time()
                prob.solve(warm_start=True)
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
        else:
            raise NotImplementedError
        return torch.tensor(sols )


###################################################################
# QCQP Problem
###################################################################
class QCQPProbem(QPProblem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   1/2 * y^T H y + G^T y <= h
                   L<= x <=U
    """
    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.H_np = dataset['H']
        self.H = torch.tensor(dataset['H'] )

    def __str__(self):
        return 'QCQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def ineq_resid(self, X, Y):
        res = []
        """
         1/2 * y^T H y + G^T y <= h
         H: m * n * n
         G: m * n
         y: 1 * n
         h: 1 * m
        """
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(Y, self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi in X_np:
                y = cp.Variable(self.ydim)
                constraints = [A @ y == Xi, y <= U, y >= L]
                for i in range(self.nineq):
                    Ht = H[i]
                    Gt = G[i]
                    ht = h[i]
                    constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  constraints)
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return sols, total_time, parallel_time

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):

        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                constraints = [A @ y == Xi, y <= U, y >= L]
                for i in range(self.nineq):
                    Ht = H[i]
                    Gt = G[i]
                    ht = h[i]
                    constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
                prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                                  constraints)
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                y.value = y_pred
                constraints = [A @ y == Xi, y <= U, y >= L]
                for i in range(self.nineq):
                    Ht = H[i]
                    Gt = G[i]
                    ht = h[i]
                    constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  constraints)
                start_time = time.time()
                prob.solve(warm_start=True)
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )


###################################################################
# SOCP Problem
###################################################################
class SOCPProbem(QPProblem):
    """
        minimize_y p^Ty
        s.t.       Ay =  x
                   ||G^T y + h||_2 <= c^Ty+d
                   L<= x <=U
    """

    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.c_np = dataset['C']
        self.d_np = dataset['d']
        self.C = torch.tensor(dataset['C'] )
        self.d = torch.tensor(dataset['d'] )

    def __str__(self):
        return 'SOCPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def ineq_resid(self, X, Y):
        res = []
        """
         ||G^T y + h||_2 <= C^Ty+d
         G: m * k * n
         h: m * k
         y: m * n
         C: m * n
         d: m * 1
        """
        q = torch.norm(torch.matmul(self.G, Y.T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(Y, self.C.T) + self.d
        res = q - p
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.c_np, self.d_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi in X_np:
                y = cp.Variable(self.ydim)
                soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(self.nineq)]
                constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y), constraints)
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return sols, total_time, parallel_time

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):

        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.c_np, self.d_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(self.nineq)]
                constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
                prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                                  constraints)
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):

        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.c_np, self.d_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                y.value = y_pred
                soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(self.nineq)]
                constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  constraints)
                start_time = time.time()
                prob.solve(warm_start=True)
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )


###################################################################
# SDP Problem
###################################################################
class SDPProbem(BaseProblem):
    """
        minimize_y tr(Qy)
        s.t.       tr(Ay) =  x
                   y >>0
                   L<= y <=U
    """
    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.Q_np = dataset['Q']
        self.A_np = dataset['A']
        self.L_np = dataset['YL']
        self.U_np = dataset['YU']
        self.Q = torch.tensor(dataset['Q'] )
        self.A = torch.tensor(dataset['A'] )
        self.xdim = dataset['X'].shape[1]
        self.ydim = (dataset['Q'].shape[0]) ** 2
        self.ymdim = dataset['Q'].shape[0]
        self.num = dataset['X'].shape[0]
        self.neq = dataset['A'].shape[0]
        self.nineq = self.ymdim
        self.nknowns = 0

        self.A = self.A.view(-1, self.ydim)
        self.Y = self.Y.permute(0, 2, 1).contiguous().view(-1, self.ydim)

        best_partial = dataset['best_partial']
        self.partial_vars = best_partial
        self.partial_unknown_vars = best_partial
        self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars)
        self.A_partial = self.A[:, self.partial_vars]
        self.A_other_inv = torch.inverse(self.A[:, self.other_vars])

        self.trainX = self.X[:-test_size]
        self.testX = self.X[-test_size:]
        self.trainY = self.Y[:-test_size]
        self.testY = self.Y[-test_size:]
        
    def __str__(self):
        return 'SDPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, Y):
        Ym = Y.view(Y.shape[0], self.ymdim, self.ymdim).permute(0, 2, 1)  # batch * n * n
        obj = torch.diagonal(torch.matmul(self.Q, Ym), dim1=-2, dim2=-1)
        return torch.sum(obj, dim=1, keepdim=True)

    def eq_resid(self, X, Y):
        return Y @ self.A.T - X

    def ineq_resid(self, X, Y):
        res = []
        """
        Y>>0 -> xYx > 0
        """
        Ym = Y.view(Y.shape[0], self.ymdim, self.ymdim).permute(0, 2, 1)  # batch * n * n

        ### sample-based methods
        num_sample = 1024
        est = torch.randn(size=(1, self.ymdim, num_sample), device=X.device)  # batch * n * k
        est = est / torch.norm(est, dim=1, p=2, keepdim=True)
        pel = torch.matmul(Ym, est)  # batch * n * k
        pel = torch.multiply(pel, est).sum(1)  # batch * n * k
        res = -1*pel.view(-1, num_sample).sum(1, keepdim=True)

        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def check_feasibility(self, X, Y):
        r1 = torch.abs(self.eq_resid(X, Y))
        r2 = torch.clamp(torch.cat([self.L - Y, Y - self.U], dim=1), 0)
        Ym = Y.view(Y.shape[0], self.ymdim, self.ymdim).permute(0, 2, 1)  # batch * n * n
        # try:
        r3 = torch.linalg.eigvals(Ym).real
        r3 = -1 * torch.min(r3, dim=1, keepdim=True)[0]
        # except:
        #     eigenvalues_list = []
        #     for i in range(Ym.shape[0]):
        #         matrix = Ym[i]
        #         eigenvalues = torch.eig(matrix, eigenvectors=False).eigenvalues[:, 0]
        #         eigenvalues_list.append(torch.min(eigenvalues).unsqueeze(0))
        #     pel = torch.stack(eigenvalues_list, dim=0)
        #     r3 = -pel.view(-1,1)
        r3 = torch.clamp(r3, 0)
        return torch.cat([r1, r2, r3], dim=1)

    def complete_partial(self, X, Y, backward=True):
        Y_full = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y_full[:, self.partial_vars] = Y
        Y_full[:, self.other_vars] = (X - Y @ self.A_partial.T) @ self.A_other_inv.T
        return Y_full

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, A, L, U = self.Q_np, self.A_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi in X_np:
                y = cp.Variable(self.ydim)
                y = cp.Variable((self.ymdim, self.ymdim), symmetric=False)
                prob = cp.Problem(cp.Minimize(cp.trace(Q @ y)),
                                  [y >> 0] + [y <= U] + [y >= L] + [cp.trace(A[i] @ y) == Xi[i] for i in
                                                                    range(self.neq)])
                prob.solve()
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return sols, total_time, parallel_time

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):

        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, A, L, U = self.Q_np, self.A_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y_pred = np.reshape(y_pred, (self.ymdim, self.ymdim)).T  # batch * n * n
                y = cp.Variable((self.ymdim, self.ymdim), symmetric=False)
                prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                                  [y >> 0] + [y <= U] + [y >= L] + [cp.trace(A[i] @ y) == Xi[i] for i in
                                                                    range(self.neq)])
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        sols = torch.tensor(sols )
        sols = sols.permute(0, 2, 1).contiguous().view(-1, self.ydim)
        return sols

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, A, L, U = self.Q_np, self.A_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y_pred = np.reshape(y_pred, (self.ymdim, self.ymdim)).T  # batch * n * n
                y = cp.Variable((self.ymdim, self.ymdim), symmetric=False)
                y.value = y_pred
                prob = cp.Problem(cp.Minimize(cp.trace(Q @ y)),
                                  [y >> 0] + [y <= U] + [y >= L] + [cp.trace(A[i] @ y) == Xi[i] for i in
                                                                    range(self.neq)])
                start_time = time.time()
                prob.solve(warm_start=True)
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        sols = torch.tensor(sols )
        sols = sols.permute(0, 2, 1).contiguous().view(-1, self.ydim)
        return sols


###################################################################
# NONCONVEX PROBLEM
###################################################################
class NonconvexProblem(QPProblem):
    """
        minimize_y 1/2 * y^T Q y + p^T sin(y)
        s.t.       Ay =  x
                   Gy <= h
    """

    def __str__(self):
        return 'NonconvexProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num))

    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1)

    def opt_solve(self, X, solver_type='ipopt', tol=1e-6):
        Q, p, A, G, h, L, U = \
            self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        total_time = 0
        for Xi in X_np:
            if solver_type == 'ipopt':
                y0 = np.linalg.pinv(A) @ Xi  # feasible initial point
                # upper and lower bounds on variables
                lb = L
                ub = U
                # upper and lower bounds on constraints
                cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
                cu = np.hstack([Xi, h])
                nlp = ipopt.problem(
                    n=len(y0),
                    m=len(cl),
                    problem_obj=nonconvex_ipopt(Q, p, A, G),
                    lb=lb,
                    ub=ub,
                    cl=cl,
                    cu=cu)
                nlp.addOption('tol', tol)
                nlp.addOption('print_level', 0)  # 3)
                start_time = time.time()
                y, info = nlp.solve(y0)
                end_time = time.time()
                Y.append(y)
                total_time += (end_time - start_time)
            else:
                raise NotImplementedError
        sols = np.array(Y)
        parallel_time = total_time / len(X_np)
        return sols, parallel_time

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-6):

        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)

                prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                                  [G @ y <= h, y <= U, y >= L,
                                   A @ y == Xi])
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )

    def opt_warmstart(self, X, Y_pred, solver_type='ipopt', tol=1e-6):
        if solver_type == 'ipopt':
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                if solver_type == 'ipopt':
                    y0 = y_pred
                    # upper and lower bounds on variables
                    lb = L
                    ub = U
                    # upper and lower bounds on constraints
                    cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
                    cu = np.hstack([Xi, h])
                    nlp = ipopt.problem(
                        n=len(y0),
                        m=len(cl),
                        problem_obj=nonconvex_ipopt(Q, p, A, G),
                        lb=lb,
                        ub=ub,
                        cl=cl,
                        cu=cu)
                    nlp.addOption('tol', tol)
                    nlp.addOption('print_level', 0)  # 3)
                    start_time = time.time()
                    y, info = nlp.solve(y0)
                    end_time = time.time()
                    Y.append(y)
                    total_time += (end_time - start_time)
                else:
                    raise NotImplementedError
        sols = np.array(Y)
        return sols

class nonconvex_ipopt(object):
    def __init__(self, Q, p, A, G):
        self.Q = Q
        self.p = p
        self.A = A
        self.G = G
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p @ np.sin(y)

    def gradient(self, y):
        return self.Q @ y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.A @ y, self.G @ y])

    def jacobian(self, y):
        return np.concatenate([self.A.flatten(), self.G.flatten()])

    # # Don't use: In general, more efficient with numerical approx
    # def hessian(self, y, lagrange, obj_factor):
    #     H = obj_factor * (self.Q - np.diag(self.p * np.sin(y)) )
    #     return H[self.tril_indices]

    # def intermediate(self, alg_mod, iter_count, obj_value,
    #         inf_pr, inf_du, mu, d_norm, regularization_size,
    #         alpha_du, alpha_pr, ls_trials):
    #     print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


###################################################################
# ACOPF
###################################################################
from pypower.api import opf, makeYbus, runpf, rundcopf, makeBdc
from pypower import idx_bus, idx_gen, idx_brch, ppoption
from pypower.idx_cost import COST

class ACOPFProblem:
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = \theta_slack   # voltage angle
                              va_ij min <= va_ij <= va_ij max
                              s_ij <= s_ij max
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """
    def __init__(self, data, test_size):
        ## Define optimization problem input and output variables
        ppc = data['ppc']
        self.ppc = ppc
        self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.baseMVA = ppc['baseMVA']
        demand = data['Dem'] / self.baseMVA
        gen = data['Gen'] / self.genbase
        voltage = data['Vol']
        self.nbus = voltage.shape[1]
        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))
        self.branch_idxes = np.concatenate([[ppc['branch'][:, idx_brch.F_BUS]], 
                                            [ppc['branch'][:, idx_brch.T_BUS]]], axis=0).T - 1
        # pv: generators wihtout slack
        # spv: generators with slack bus (slack bus with known vol angle)
        # pq: load bus (zero Pg Qg generation)
        # ng = len(spv), npv = len(pv), nslack = len(slack), nbus = ng + len(pq)
        # indices within generator
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.nb = ppc['bus'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)

        self.quad_costs = torch.tensor(ppc['gencost'][:, 4] )
        self.lin_costs = torch.tensor(ppc['gencost'][:, 5] )
        self.const_cost = ppc['gencost'][:, 6].sum()

        # initial values for solver
        self.pg_init = torch.tensor(ppc['gen'][:, idx_gen.PG] / self.genbase)
        self.qg_init = torch.tensor(ppc['gen'][:, idx_gen.QG] / self.genbase)
        self.vm_init = torch.tensor(ppc['bus'][:, idx_bus.VM])
        self.va_init = torch.tensor(np.deg2rad(ppc['bus'][:, idx_bus.VA]))
        # upper and lower bound
        self.pmax = torch.tensor(ppc['gen'][:, idx_gen.PMAX] / self.genbase )
        self.pmin = torch.tensor(ppc['gen'][:, idx_gen.PMIN] / self.genbase )
        self.qmax = torch.tensor(ppc['gen'][:, idx_gen.QMAX] / self.genbase )
        self.qmin = torch.tensor(ppc['gen'][:, idx_gen.QMIN] / self.genbase )
        self.vmax = torch.tensor(ppc['bus'][:, idx_bus.VMAX] )
        self.vmin = torch.tensor(ppc['bus'][:, idx_bus.VMIN] )
        self.smax = torch.tensor(ppc['branch'][:, idx_brch.RATE_A] / self.baseMVA) 
        self.amax = torch.tensor(np.deg2rad(ppc['branch'][:, idx_brch.ANGMAX])) 
        self.amin = torch.tensor(np.deg2rad(ppc['branch'][:, idx_brch.ANGMIN]))
        self.slackva = self.va_init[self.slack]
        # torch.tensor(np.array([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])])).squeeze(-1)


        ppc2 = copy.deepcopy(ppc)
        ppc2['bus'][:, 0] -= 1
        ppc2['branch'][:, [0, 1]] -= 1
        Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus) )
        self.Ybusi = torch.tensor(np.imag(Ybus) )

        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2 * self.ng
        self.va_start_yidx = 2 * self.ng + self.nbus

        ## Keep parameters indicating how data was generated
        self.EPS_INTERIOR = data['EPS_INTERIOR'][0][0]
        self.CorrCoeff = data['CorrCoeff'][0][0]
        self.MaxChangeLoad = data['MaxChangeLoad'][0][0]
        base_load = torch.tensor(np.concatenate([ppc['bus'][:, idx_bus.PD],
                                                 ppc['bus'][:, idx_bus.QD]],axis=0)) / self.baseMVA
        self.input_L = base_load * (1-self.MaxChangeLoad)
        self.input_U = base_load * (1+self.MaxChangeLoad)

        ## Define train/valid/test split
        # self.valid_frac = valid_frac
        # self.test_frac = test_frac

        ### Load data
        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)
        Y = np.concatenate([np.real(gen), np.imag(gen), 
                            np.abs(voltage), np.angle(voltage)], axis=1)
        feas_mask = ~np.isnan(Y).any(axis=1)
        self.X = torch.tensor(X[feas_mask] )
        self.Y = torch.tensor(Y[feas_mask] )
        self.xdim = X.shape[1]
        self.ydim = Y.shape[1]
        self.num = feas_mask.sum()
        self.neq = 2 * self.nbus
        self.nineq = 4 * self.ng + 2 * self.nbus
        self.nknowns = self.nslack
        print(self.neq, self.nineq, self.ydim, self.xdim)

        self.trainX = self.X[:-test_size]
        self.testX = self.X[-test_size:]
        self.trainY = self.Y[:-test_size]
        self.testY = self.Y[-test_size:]


        ## Define variables and indices for "partial completion" neural network
        # pg (non-slack) and |v|_g (including slack)
        self.partial_vars = np.concatenate([self.pg_start_yidx + self.pv_, 
                                            self.vm_start_yidx + self.spv, 
                                            self.va_start_yidx + self.slack])
        self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars)
        self.partial_unknown_vars = np.concatenate([self.pg_start_yidx + self.pv_, 
                                                    self.vm_start_yidx + self.spv])

        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2 * self.npv + self.nslack)
        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus

        ### For Pytorch
        self.device = DEVICE



    def __str__(self):
        return 'ACOPF-{}-{}-{}-{}-{}-{}'.format(
            self.nbus, self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,  0.2, 0.0)

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2 * self.ng]
        vm = Y[:, -2 * self.nbus:-self.nbus]
        va = Y[:, -self.nbus:]
        return pg, qg, vm, va

    def obj_fn(self, Y):
        pg, _, _, _ = self.get_yvars(Y)
        pg_mw = pg * torch.tensor(self.genbase ).to(Y.device)
        cost = (self.quad_costs * pg_mw ** 2).sum(axis=1) + \
               (self.lin_costs * pg_mw).sum(axis=1) + \
               self.const_cost
        return cost / (self.genbase.mean() ** 2)

    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)

        ## power balance equations
        tmp1 = vr @ self.Ybusr - vi @ self.Ybusi
        tmp2 = -vr @ self.Ybusi - vi @ self.Ybusr

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=X.device)
        pg_expand[:, self.spv] = pg
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr * tmp1 - vi * tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=X.device)
        qg_expand[:, self.spv] = qg
        react_resid = (qg_expand - X[:, self.nbus:]) - (vr * tmp2 + vi * tmp1)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)

        return resids

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        ## Bus llimit
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm,
            # self.branch_ineq_resid(X, Y)
        ], dim=1)
        return torch.clamp(resids, 0)

    def branch_ineq_resid(self, X, Y):
        _, _, vm, va = self.get_yvars(Y)
        ### Branch angele limit
        vaij = va.view(-1,self.nbus,1) - va.view(-1,1,self.nbus)
        branch_a = vaij[:,self.branch_idxes[:,0], self.branch_idxes[:,1]]
        resids_brach_angle = torch.cat([branch_a - self.amax, 
                                  self.amin - branch_a], dim=1)
        
        ### Branch flow limit
        vmij = vm.view(-1,self.nbus,1) * vm.view(-1,1,self.nbus)
        vmii = vm.view(-1,self.nbus,1) ** 2
        sin_vaij = torch.sin(vaij)
        cos_vaij = torch.cos(vaij)
        pij = (
            self.Ybusr.view(-1,self.nbus,self.nbus) * vmii -  \
            vmij * (self.Ybusi.view(-1,self.nbus,self.nbus)*sin_vaij + \
            self.Ybusr.view(-1,self.nbus,self.nbus)*cos_vaij)
            )[:,self.branch_idxes[:,0], self.branch_idxes[:,1]]
        qij = (
            - self.Ybusi.view(-1,self.nbus,self.nbus) * vmii -  \
            vmij * (self.Ybusr.view(-1,self.nbus,self.nbus)*sin_vaij - \
            self.Ybusi.view(-1,self.nbus,self.nbus)*cos_vaij)
            )[:,self.branch_idxes[:,0], self.branch_idxes[:,1]]
        sij = torch.sqrt(pij**2 + qij**2)
        resids_brach_flow = sij - self.smax 
        # print(sij, self.smax ** 2)
        # print(1/0)
        return torch.clamp(torch.cat([resids_brach_angle, resids_brach_flow], dim=1), 0)

    def eq_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        eq_resid = self.eq_resid(X, Y)
        return 2 * eq_jac.transpose(1, 2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    def ineq_grad(self, X, Y, mode='unfold'):
        if mode == 'unfold':
            ineq_jac = self.ineq_jac(Y)
            ineq_resid = self.ineq_resid(X, Y)
            return 2 * ineq_jac.transpose(1, 2).bmm(ineq_resid.unsqueeze(-1)).squeeze(-1)
        elif mode == 'autograd':
            grad_list = []
            for n in range(Y.shape[0]):
                x = X[n].view(1, -1)
                y = Y[n].view(1, -1)
                y = torch.autograd.Variable(y, requires_grad=True)
                ineq_penalty = self.ineq_resid(x, y) ** 2
                ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
                grad = torch.autograd.grad(ineq_penalty, y)[0]
                grad_list.append(grad.view(1, -1))
            grad = torch.cat(grad_list, dim=0)
            return grad        

    def ineq_partial_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])

        direct_grad = self.ineq_grad(X, Y)
        indirect_partial_grad = dynz_dz.transpose(1, 2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars]

        full_grad = torch.zeros(X.shape[0], self.ydim, device=X.device)
        full_grad[:, self.partial_vars] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)

        return full_grad

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)
        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(
            torch.multiply(v1, v2))  # torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        Ydiagv = lambda Y, v: torch.multiply(Y.unsqueeze(0), v.unsqueeze(
            1))  # Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        dtm = lambda v, M: torch.multiply(v.unsqueeze(2), M)  # torch.diag_embed(v).bmm(M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        Yr = self.Ybusr
        Yi = self.Ybusi
        YrvrYivi = vr @ Yr - vi @ Yi
        YivrYrvi = vr @ Yi + vi @ Yr
        # print(cosva.shape, YrvrYivi.shape, Yi.shape, Ydiagv(Yr, -vi).shape)
        # print(1/0)
        # real power equations
        dreal_dpg = torch.zeros(self.nbus, self.ng, device=Y.device)
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=Y.device)
        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva) - Ydiagv(Yi, sinva)) \
                    - mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva) + Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi) - Ydiagv(Yi, vr)) \
                    - mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi) + Ydiagv(Yr, vr))

        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ng, device=Y.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=Y.device)
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva) + Ydiagv(Yr, sinva)) \
                     - mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva) - Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi) + Ydiagv(Yr, vr)) \
                     - mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi) - Ydiagv(Yi, vr))

        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape),
                       torch.zeros(vr.shape[0], self.nbus, self.ng, device=Y.device),
                       dreal_dvm, dreal_dva], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ng, device=Y.device),
                       dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                       dreact_dvm, dreact_dva], dim=2)], dim=1)
        return jac

    def ineq_jac(self, Y):
        jac = torch.cat([
            torch.cat([torch.eye(self.ng, device=Y.device),
                       torch.zeros(self.ng, self.ng, device=Y.device),
                       torch.zeros(self.ng, self.nbus, device=Y.device),
                       torch.zeros(self.ng, self.nbus, device=Y.device)], dim=1),
            torch.cat([-torch.eye(self.ng, device=Y.device),
                       torch.zeros(self.ng, self.ng, device=Y.device),
                       torch.zeros(self.ng, self.nbus, device=Y.device),
                       torch.zeros(self.ng, self.nbus, device=Y.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=Y.device),
                       torch.eye(self.ng, device=Y.device),
                       torch.zeros(self.ng, self.nbus, device=Y.device),
                       torch.zeros(self.ng, self.nbus, device=Y.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=Y.device),
                       -torch.eye(self.ng, device=Y.device),
                       torch.zeros(self.ng, self.nbus, device=Y.device),
                       torch.zeros(self.ng, self.nbus, device=Y.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=Y.device),
                       torch.zeros(self.nbus, self.ng, device=Y.device),
                       torch.eye(self.nbus, device=Y.device),
                       torch.zeros(self.nbus, self.nbus, device=Y.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=Y.device),
                       torch.zeros(self.nbus, self.ng, device=Y.device),
                       -torch.eye(self.nbus, device=Y.device),
                       torch.zeros(self.nbus, self.nbus, device=Y.device)], dim=1)
        ], dim=0)
        return jac.unsqueeze(0).expand(Y.shape[0], *jac.shape)

    def scale_full(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        pg = pg * (self.pmax - self.pmin) + self.pmin
        qg = qg * (self.qmax - self.qmin) + self.qmin
        vm = vm * (self.vmax - self.vmin) + self.vmin
        va = va * 2 * np.pi - np.pi # (-pi, pi)
        va[:, self.slack] = self.slack_va#.unsqueeze(0).expand(X.shape[0], self.nslack)
        return torch.cat([pg, qg, vm, va], dim=1)

    def scale_partial(self, X, Y):
        Y_scaled = torch.zeros(Y.shape, device=Y.device)
        # Re-scale real powers
        # print(Z[:, self.pg_pv_zidx])
        # print(self.pmax.shape, self.spv)
        Y_scaled[:, self.pg_pv_zidx] = Y[:, self.pg_pv_zidx] * (self.pmax[self.pv_] - self.pmin[self.pv_]) + self.pmin[
            self.pv_]
        # Re-scale voltage magnitudes
        Y_scaled[:, self.vm_spv_zidx] = Y[:, self.vm_spv_zidx] * (self.vmax[self.spv] - self.vmin[self.spv]) + \
                                        self.vmin[self.spv]
        return Y_scaled

    def scale(self, X, Y):
        if Y.shape[1] == len(self.partial_unknown_vars):
            Y_scale = self.scale_partial(X, Y)
        else:
            Y_scale = self.scale_full(X, Y)
        return Y_scale

    def complete_partial(self, X, Y, backward=True):
        if backward:
            return PFFunction(self)(X, Y)
        else:
            with torch.no_grad():
                return PFFunction(self)(X, Y)

    def cal_penalty(self, X, Y):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y):
        return self.cal_penalty(X, Y)
    
    def opt_solve(self, X, solver_type='pypower', tol=1e-5):
        X_np = X.detach().cpu().numpy()
        ppc = self.ppc

        # Set reduced voltage bounds if applicable
        ppc['bus'][:, idx_bus.VMIN] = ppc['bus'][:, idx_bus.VMIN] + self.EPS_INTERIOR
        ppc['bus'][:, idx_bus.VMAX] = ppc['bus'][:, idx_bus.VMAX] - self.EPS_INTERIOR

        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
        Y = []
        total_time = 0
        for i in range(X_np.shape[0]):
            # print(X_np[i, :self.nbus] * self.baseMVA)
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA
            # print(ppc.items())
            start_time = time.time()
            my_result = opf(ppc, ppopt)
            end_time = time.time()
            total_time += (end_time - start_time)
            print(i, end='\r')
            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))

        return np.array(Y), total_time, total_time / len(X_np)

    def opt_proj(self, X, Y, solver_type='pypower', tol=1e-5):
        pg, qg, vm, va = self.get_yvars(Y)
        pg_all = pg.detach().cpu().numpy() * self.genbase
        qg_all = qg.detach().cpu().numpy() * self.genbase
        vm_all = vm.detach().cpu().numpy()
        va_all = np.rad2deg(va.detach().cpu().numpy())
        X_np = X.detach().cpu().numpy()
        Y = []
        total_time = 0
        start_time = time.time()
        for i in range(X_np.shape[0]):
            print(i, end='\r')
            pg_0 = pg_all[i]
            pd = X_np[i]
            ppc = copy.deepcopy(self.ppc)
            ppc['gencost'][:, COST] = 1
            ppc['gencost'][:, COST + 1] = -2 * pg_0
            # Set reduced voltage bounds if applicable
            ppc['bus'][:, idx_bus.VMIN] = ppc['bus'][:, idx_bus.VMIN] + self.EPS_INTERIOR
            ppc['bus'][:, idx_bus.VMAX] = ppc['bus'][:, idx_bus.VMAX] - self.EPS_INTERIOR
            ppc['bus'][:, idx_bus.VM] = vm_all[i]
            ppc['bus'][:, idx_bus.VA] = va_all[i]
            ppc['gen'][:, idx_gen.PG] = pg_all[i]
            ppc['gen'][:, idx_gen.QG] = qg_all[i]
            # Solver 1
            ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
            ppc['bus'][:, idx_bus.PD] = pd[:self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = pd[self.nbus:] * self.baseMVA
            my_result = opf(ppc, ppopt)
            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))
        end_time = time.time()
        total_time += (end_time - start_time)
        return torch.tensor(np.array(Y) )

    def opt_warmstart(self, X, Y, solver_type='pypower', tol=1e-5):
        pg, qg, vm, va = self.get_yvars(Y)
        pg_all = pg.detach().cpu().numpy() * self.genbase
        qg_all = qg.detach().cpu().numpy() * self.genbase
        vm_all = vm.detach().cpu().numpy()
        va_all = np.rad2deg(va.detach().cpu().numpy())
        X_np = X.detach().cpu().numpy()
        Y = []
        total_time = 0
        start_time = time.time()
        for i in range(X_np.shape[0]):
            print(i, end='\r')
            ppc = copy.deepcopy(self.ppc)
            pd = X_np[i]
            # Set reduced voltage bounds if applicable
            ppc['bus'][:, idx_bus.VMIN] = ppc['bus'][:, idx_bus.VMIN] + self.EPS_INTERIOR
            ppc['bus'][:, idx_bus.VMAX] = ppc['bus'][:, idx_bus.VMAX] - self.EPS_INTERIOR

            ppc['bus'][:, idx_bus.VM] = vm_all[i]
            ppc['bus'][:, idx_bus.VA] = va_all[i]
            ppc['gen'][:, idx_gen.PG] = pg_all[i]
            ppc['gen'][:, idx_gen.QG] = qg_all[i]
            # Solver options
            ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM

            ppc['bus'][:, idx_bus.PD] = pd[:self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = pd[self.nbus:] * self.baseMVA

            my_result = opf(ppc, ppopt)
            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))
        end_time = time.time()
        total_time += (end_time - start_time)
        return torch.tensor(np.array(Y) )

def PFFunction(data, tol=1e-5, bsz=1024, max_iters=10):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            # start_time = time.time()
            ## Step 1: Newton's method
            Y = torch.zeros(X.shape[0], data.ydim, device=X.device)
            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            Y[:, data.pg_start_yidx + data.pv_] = Z[:, data.pg_pv_zidx]  # pg at non-slack gens
            Y[:, data.vm_start_yidx + data.spv] = Z[:, data.vm_spv_zidx]  # vm at gens
            # init guesses for remaining values
            Y[:, data.vm_start_yidx + data.pq] = data.vm_init[data.pq]  # vm at load buses
            Y[:, data.va_start_yidx: data.va_start_yidx+data.nb] = data.va_init   # va at all bus
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = 0  # qg at gens (not used in Newton upd)
            Y[:, data.pg_start_yidx + data.slack_] = 0  # pg at slack (not used in Newton upd)

            keep_constr = np.concatenate([
                data.pflow_start_eqidx + data.pv,  # real power flow at non-slack gens
                data.pflow_start_eqidx + data.pq,  # real power flow at load buses
                data.qflow_start_eqidx + data.pq])  # reactive power flow at load buses
            newton_guess_inds = np.concatenate([
                data.vm_start_yidx + data.pq,  # vm at load buses
                data.va_start_yidx + data.pv,  # va at non-slack gens
                data.va_start_yidx + data.pq])  # va at load buses

            converged = torch.zeros(X.shape[0])
            jacs = []
            # newton_jacs_inv = []
            for b in range(0, X.shape[0], bsz):
                X_b = X[b:b + bsz]
                Y_b = Y[b:b + bsz]
                for _ in range(max_iters):
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    jac_full = data.eq_jac(Y_b)
                    jac = jac_full[:, keep_constr, :]
                    jac = jac[:, :, newton_guess_inds]
                    # start_time = time.time()
                    """Direct inverse"""
                    # newton_jac_inv = torch.inverse(jac)
                    # delta = torch.matmul(newton_jac_inv, gy.unsqueeze(-1)).squeeze(-1)
                    """LU decomposition"""
                    # jac_lu = torch.linalg.lu_factor(jac)
                    # delta = torch.linalg.ldl_solve(jac_lu, gy.unsqueeze(-1)).squeeze(-1)
                    """Linear system"""
                    delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)
                    """Approximation"""
                    # delta = 0
                    # tt = torch.eye(jac.shape[-1]).to(jac.device) - 0.01 * jac
                    # jac_inv = gy
                    # for _ in range(5):
                    #     jac_inv = torch.matmul(tt, jac_inv.unsqueeze(-1)).squeeze(-1)
                    #     delta += jac_inv
                    # delta = 0.01 * delta
                    # print('lin run_time', time.time()-start_time)
                    # ineq_step = data.ineq_grad(X_b, Y_b)
                    Y_b[:, newton_guess_inds] -= delta
                    if torch.abs(delta).max() < tol:
                        break
                if torch.abs(delta).max() > tol:
                    print('Newton methods for Power Flow does not converge')
                # print(torch.abs(delta).max())
                converged[b:b + bsz] = (delta.abs() < tol).all(dim=1)
                jacs.append(jac_full)
                # newton_jacs_inv.append(newton_jac_inv)

            ## Step 2: Solve for remaining variables
            # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            Y[:, data.pg_start_yidx + data.slack_] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.slack]

            ctx.data = data
            ctx.save_for_backward(torch.cat(jacs),
                                  torch.as_tensor(newton_guess_inds, device=X.device),
                                  torch.as_tensor(keep_constr, device=X.device))
            return Y

        @staticmethod
        def backward(ctx, dl_dy):

            data = ctx.data
            # jac, newton_jac_inv, newton_guess_inds, keep_constr = ctx.saved_tensors
            jac, newton_guess_inds, keep_constr = ctx.saved_tensors

            ## Step 2 (calc pg at slack and qg at gens)
            jac_pre_inv = jac[:, keep_constr, :]
            jac_pre_inv = jac_pre_inv[:, :, newton_guess_inds]

            # gradient of all voltages through step 3 outputs
            last_eqs = np.concatenate([data.pflow_start_eqidx + data.slack, data.qflow_start_eqidx + data.spv])
            last_vars = np.concatenate([
                data.pg_start_yidx + data.slack_, np.arange(data.qg_start_yidx, data.qg_start_yidx + data.ng)])
            jac3 = jac[:, last_eqs, :]
            dl_dvmva_3 = -jac3[:, :, data.vm_start_yidx:].transpose(1, 2).bmm(
                dl_dy[:, last_vars].unsqueeze(-1)).squeeze(-1)

            # gradient of pd at slack and qd at gens through step 3 outputs
            dl_dpdqd_3 = dl_dy[:, last_vars]

            # insert into correct places in x and y loss vectors
            dl_dy_3 = torch.zeros(dl_dy.shape, device=jac.device)
            dl_dy_3[:, data.vm_start_yidx:] = dl_dvmva_3

            dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=jac.device)
            dl_dx_3[:, np.concatenate([data.slack, data.nbus + data.spv])] = dl_dpdqd_3

            ## Step 1
            dl_dy_total = dl_dy_3 + dl_dy  # Backward pass vector including result of last step

            # Use precomputed inverse jacobian
            jac2 = jac[:, keep_constr, :]
            # d_int = newton_jac_inv.transpose(1, 2).bmm(dl_dy_total[:, newton_guess_inds].unsqueeze(-1)).squeeze(-1)
            d_int = torch.linalg.solve(jac_pre_inv.transpose(1, 2),
                                       dl_dy_total[:, newton_guess_inds].unsqueeze(-1)).squeeze(-1)

            dl_dz_2 = torch.zeros(dl_dy.shape[0], data.npv + data.ng, device=jac.device)
            dl_dz_2[:, data.pg_pv_zidx] = -d_int[:, :data.npv]  # dl_dpg at pv buses
            dl_dz_2[:, data.vm_spv_zidx] = -jac2[:, :, data.vm_start_yidx + data.spv].transpose(1, 2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)

            dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=jac.device)
            dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
            dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
            dl_dx_2[:, data.nbus + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses

            # Final quantities
            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                data.pg_start_yidx + data.pv_, data.vm_start_yidx + data.spv])]
            return dl_dx_total, dl_dz_total
    return PFFunctionFn.apply

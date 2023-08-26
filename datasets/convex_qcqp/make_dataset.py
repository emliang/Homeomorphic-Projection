import numpy as np
import pickle
import torch
import sys
import os
import cvxpy as cp
import os


num_var = 100
num_ineq = 50
num_eq = 50
num_examples = 10000

np.random.seed(2023)
Q = np.diag(np.random.rand(num_var)*0.5)
p = np.random.uniform(-1, 1, num_var)
A = np.random.uniform(-1, 1, size=(num_eq, num_var))
X = np.random.uniform(-0.5, 0.5, size=(num_examples, num_eq))
G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)
H = np.random.uniform(0, 0.1,  size=(num_ineq, num_var))
H = [np.diag(H[i]) for i in range(num_ineq)]
H = np.array(H)
L = np.ones((num_var)) * -5
U = np.ones((num_var)) * 5
data = {'Q':Q,
        'p':p,
        'A':A,
        'X':X,
        'G':G,
        'H':H,
        'h':h,
        'L':L,
        'U':U,
        'Y':[]}
Y = []
for n in range(num_examples):
    Xi = X[n]
    y = cp.Variable(num_var)
    constraints = [A @ y == Xi,y <= U, y >= L]
    for i in range(num_ineq):
        Ht = H[i]
        Gt = G[i]
        ht = h[i]
        constraints.append(0.5*cp.quad_form(y,Ht) + Gt.T @ y <= ht)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                        constraints)
    prob.solve()
    print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    n+=1
    Y.append(y.value)

data['Y'] = np.array(Y)


i = 0
det_min = 0
best_partial = 0
while i < 1000:
    np.random.seed(i)
    partial_vars = np.random.choice(num_var, num_var - num_eq, replace=False)
    other_vars = np.setdiff1d(np.arange(num_var), partial_vars)
    _, det = np.linalg.slogdet(A[:, other_vars])
    if det>det_min:
        det_min = det
        best_partial = partial_vars
    i += 1
print('best_det', det_min)
data['best_partial'] = best_partial



with open("datasets/convex_qcqp/random_convex_qcqp_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(data, f)

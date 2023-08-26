import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from optimization_utils import nonconvex_ipopt
import ipopt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

num_var = 200
num_ineq = 100
num_eq = 100
num_examples = 20000



print(num_ineq, num_eq)
np.random.seed(2023)
Q = np.diag(np.random.rand(num_var)*0.5)
p = np.random.uniform(-1, 1, num_var)
A = np.random.uniform(-1, 1, size=(num_eq, num_var))
X = np.random.uniform(-0.5, 0.5, size=(num_examples, num_eq))
G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)
L = np.ones((num_var))*-5
U = np.ones((num_var))*5
data = {'Q':Q,
        'p':p,
        'A':A,
        'X':X,
        'G':G,
        'h':h,
        'L':L,
        'U':U,
        'Y':[]}

Y = []
for n in range(num_examples):
    Xi = X[n]
    y0 = np.linalg.pinv(A)@Xi  # feasible initial point
    cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
    cu = np.hstack([Xi, h])
    nlp = ipopt.problem(
                n=len(y0),
                m=len(cl),
                problem_obj=nonconvex_ipopt(Q, p, A, G),
                lb=L,
                ub=U,
                cl=cl,
                cu=cu)
    nlp.addOption('tol', 1e-5)
    nlp.addOption('print_level', 0) # 3)
    y, info = nlp.solve(y0)
    Y.append(y)
    print(n, np.max(y), np.min(y), y[0:5].T, end='\r')
    n+=1
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




with open("datasets/nonconvex/random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(data, f)

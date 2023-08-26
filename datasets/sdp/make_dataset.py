import numpy as np
import pickle
import cvxpy as cp


num_var = 15 # n*n
num_eq = 100
num_ineq = 100
num_examples = 20000



print(num_ineq, num_eq)
np.random.seed(2022)
Q = np.random.uniform(-1,1, size=(num_var, num_var))
A = []
X = np.random.uniform(-0.5, 0.5, size=(num_examples, num_eq))
for i in range(num_eq):
    A.append(np.random.uniform(-1,1, size=(num_var, num_var)))
L = np.ones((num_var, num_var))*-5
U = np.ones((num_var, num_var))*5


p = np.random.uniform(-1, 1, num_var)
G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)


data = {'Q':Q,
        'p':p,
        'A':np.array(A),
        'X':X,
        'G':G,
        'h':h,
        'L':L,
        'U':U,
        'Y':[]}

Y = []
for n in range(num_examples):
    Xi = X[n]
    y = cp.Variable((num_var,num_var), symmetric=False)
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ y)),
                        [y>>0]+[y<=U]+[y>=L]+[cp.trace(A[i] @ y) == Xi[i] for i in range(num_eq)])
    prob.solve()
    print(n, np.max(y.value), np.min(y.value), y.value[0:5,0].T, end='\r')
    n+=1
    Y.append(y.value)
data['Y'] = np.array(Y)




i = 0
det_min = 0
best_partial = 0
A_extend = np.array([A[i].flatten() for i in range(num_eq)])
while i < 1000:
    np.random.seed(i)
    partial_vars = np.random.choice(num_var**2, num_var**2 - num_eq, replace=False)
    other_vars = np.setdiff1d(np.arange(num_var**2), partial_vars)
    _, det = np.linalg.slogdet(A_extend[:, other_vars])
    if det>det_min:
        det_min = det
        best_partial = partial_vars
    i += 1
print('best_det', det_min)
data['best_partial'] = best_partial


with open("datasets/sdp/random_sdp_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var**2, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(data, f)

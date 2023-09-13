import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import time
import os
from training_all import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
defaults = config()

def main():
    args = config()
    for prob in ['qp']:
        if prob == 'acopf':
            for size in [[30, 10000], 
                         [118, 20000]]:
                for proj in ['H_Bis']:
                    args['opfSize'] = size
                    args['projType'] = proj
                    args['probType'] = prob
                    test_single(args)
        else:
            for size in [[200, 100, 100, 20000]]:
                for proj in ['H_Bis']:
                    args['probSize'] = size
                    args['projType'] = proj
                    args['probType'] = prob
                    test_single(args)
                    



def test_single(args):
    data, result_save_dir, model_save_dir = load_instance(args)
    test_nn_solver(data, args, model_save_dir, result_save_dir)
    test_inf_time(data, args, model_save_dir, result_save_dir)




if __name__ == '__main__':
    main()

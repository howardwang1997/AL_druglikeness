import os
import sys
import numpy as np
import pandas as pd
import torch

import split_data as split
import preprocessing
import NN
import active

from alipy.query_strategy import (QueryInstanceQBC, QueryInstanceGraphDensity,
                                  QueryInstanceUncertainty, QueryRandom,
                                  QueryCostSensitiveHALC, QueryCostSensitivePerformance,
                                  QueryCostSensitiveRandom)
from alipy import index
from alipy.experiment import StoppingCriteria

from alipy import ToolBox
import copy
import joblib

import pandas as pd
import numpy as np
import torch

from rdkit.Chem import AllChem
from rdkit import Chem

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors as rdmd

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from math import sqrt

import warnings
warnings.filterwarnings('ignore')

from time import clock


def _label(path):
    suffix = path.split('.')[-1]
    if suffix == 'txt':
        fd = open(path, 'r+')
        lab = fd.read().split('')
        fd.close()
        lab = np.array([float(i) for i in lab])
    else: lab = np.array(joblib.load(path))
    return lab


def _dataset(path):
    suffix = path.split('.')[-1]
    set = None
    if suffix == 'txt' or suffix == 'smi':
        fd = open(path, 'r+')
        set = fd.read().split('')
        fd.close()
        set = preprocessing.smile_list_to_mols(set)
    elif suffix == 'sdf':
        set = preprocessing.read_sdf(path)

    if set:
        set = preprocessing.descriptors.generate_rdDescriptorsSets(set)
    else:
        if suffix == 'csv':
            set = np.array(pd.read_csv(path))
        else: set = np.array(joblib.load(path))

    return set


def kfold(ds, lab, k=5, regression=False, model=None):
    # torch5Fold
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True)

    fold = 0
    for tr_idx, te_idx in kf.split(lab):
        try:
            os.mkdir('active%d' % fold)
        except FileExistsError:
            pass
        path = 'active%d' % fold
        fold += 1

        xtr, ytr, xte, yte = ds[tr_idx], lab[tr_idx], ds[te_idx], lab[te_idx]
        if model: pass
        else:
            if regression: model = NN.FcRegRDKit()
            else: model = NN.FcRDKit()
        print('start training')
        if regression: nn = active.TorchRegressionFold(xtr, ytr, xte, yte, model, 'active', path, ['percent_of_unlabel', 1],
                                      measure='distance', distance='linear')

        else: nn = active.TorchFold(xtr, ytr, xte, yte, model, 'active', path, ['percent_of_unlabel', 1])
        nn.train()
    print('finish training')


def split_train(ds, lab, test_ratio=0.3, regression=False, model=None):
    all = len(lab)
    splitor = split.TTSplit(all, 'portion', test=test_ratio)
    tr_idx, te_idx = splitor.split()
    xtr, ytr, xte, yte = ds[tr_idx], lab[tr_idx], ds[te_idx], lab[te_idx]

    if model: pass
    else:
        if regression: model = NN.FcRegRDKit()
        else: model = NN.FcRDKit()
    print('start training')
    if regression: nn = active.TorchRegressionFold(xtr, ytr, xte, yte, model, 'active', path, ['percent_of_unlabel', 1],
                                  measure='distance', distance='linear')

    else: nn = active.TorchFold(xtr, ytr, xte, yte, model, 'active', path, ['percent_of_unlabel', 1])
    nn.train()
    print('finish training')


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args = sys.argv
args.pop(0)
vari = {}
fold = False
regression = False
trained = False
# for i in range(int(len(args)/2)):
for i in range(1):
    if '-d' in args:
        path_trainset = args[args.index('-d')+1]
        trainset = _dataset(path_trainset)
    elif '--dataset' in args:
        path_trainset = args[args.index('--dataset')+1]
        trainset = _dataset(path_trainset)

    if '-l' in args:
        path_trainlab = args[args.index('-l')+1]
        trainlab = _label(path_trainlab)
    elif '--labels' in args:
        path_trainlab = args[args.index('--labels')+1]
        trainlab = _label(path_trainlab)

    if '-s' in args:
        path_save = args[args.index('-s')+1]
    elif '--save' in args:
        path_save = args[args.index('--save')+1]

    if '-f' in args:
        k = int(args[args.index('-f')+1])
        fold = True
    elif '--fold' in args:
        k = int(args[args.index('--fold')+1])
        fold = True
    elif '-r' in args:
        test_ratio = float(args[args.index('-r')+1])
    elif '--ratio' in args:
        test_ratio = float(args[args.index('--ratio')+1])

    if '-R' in args:
        regression = True
    if '--regression' in args:
        regression = True

    if '-m' in args:
        path_model = args[args.index('-m')+1]
        model = torch.load(path_model)
        trained = True
    elif '--model' in args:
        path_model = args[args.index('--model')+1]
        model = torch.load(path_model)
        trained = True


# start training from here


file_path = path_save
try:
    os.mkdir(file_path)
except FileExistsError:
    pass
os.chdir(file_path)


trainset, trainlab = torch.tensor(trainset), torch.tensor(trainlab)

if fold:
    if trained: kfold(trainset, trainlab, k=k, regression=regression, model=model)
    else: kfold(trainset, trainlab, k=k, regression=regression)
else:
    if trained: split_train(trainset, trainlab, test_ratio=test_ratio, regression=regression, model=model)
    else: split_train(trainset, trainlab, test_ratio=test_ratio, regression=regression)

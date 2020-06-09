from alipy.query_strategy import (QueryInstanceQBC, QueryInstanceGraphDensity,
                                  QueryInstanceUncertainty, QueryRandom,
                                  QueryCostSensitiveHALC, QueryCostSensitivePerformance,
                                  QueryCostSensitiveRandom)
from alipy import index
from alipy.experiment import StoppingCriteria

from alipy import ToolBox
import copy
import joblib

import split_data as split
import preprocessing
import CNN

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

import AL
from time import clock
import os

# CHANGE HERE
phase = 'train'
# phase = 'test'

if phase == 'train':
    # CHANGE HERE
    trainset = joblib.load('rdtr_dataset')
    trainlab = joblib.load(('rdtr_lab'))


    trainset, trainlab = torch.tensor(trainset), torch.tensor(trainlab)

    # torch
    st = clock()

    try:
        # CHANGE HERE
        os.mkdir('rdal0.005')
    except FileExistsError:
        pass
    # CHANGE HERE
    path = 'rdal0.005'

    model = CNN.FcRDKit()
    print('start training')
    clf = AL.Torch(trainset, trainlab, 0.2, model, path, ['percent_of_unlabel', 1])
    clf.classify()
    print('time:', clock()-st)

elif phase == 'test':
    model = ''
    model = torch.load(model)
    net = CNN.NeuralNetwork(model=model)

    # inport data in format .sdf
    # CHANGE HERE
    file = ''
    mols = preprocessing.mols_to_smiles_list(preprocessing.read_sdf(file))
    dataset = [preprocessing.descriptors.generate_rdDescriptors(mol) for mol in mols]
    del mols
    dataset = torch.tensor(dataset)

    net.predict(dataset)
    preds = net.preds

    # CHANGE HERE
    joblib.dump(preds, 'prediction')

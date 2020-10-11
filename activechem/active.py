from alipy.query_strategy import (QueryInstanceQBC, QueryInstanceGraphDensity,
                                  QueryInstanceUncertainty,
                                  QueryCostSensitiveHALC, QueryCostSensitivePerformance,
                                  QueryCostSensitiveRandom)
from alipy.query_strategy import QueryInstanceRandom as QueryRandom
from alipy import index
from alipy.experiment import StoppingCriteria

from alipy import ToolBox
import copy
import joblib

import split_data as split
import preprocessing
import NN
from query_regression import QueryInstanceDistribution, QueryInstanceResidueRegressor

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


class TorchFold:
    def __init__(self, dataset, labels, testset, testlab, model, phase, path, stopping):
        self.dataset = dataset
        self.labels = labels
        self.testset = testset
        self.testlab = testlab
        self.model = model
        self.phase = phase
        self.classes = int(max(labels))
        self.alibox = ToolBox(X=dataset, y=labels, query_type='AllLabels', saving_path='./%s' % path)
        self.alibox.split_AL(test_ratio=0, initial_label_rate=0.05, split_count=1)
        self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
        self.query_strategy = QueryInstanceUncertainty(X=dataset, y=labels, measure='least_confident')
        # self.query_strategy = QueryInstanceQBC(disagreement='KL_divergence')
        self.random = QueryRandom()
        self.unc_result = []
        self.title = ''
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.specificity = []
        self.auc = []
        self.f1 = []
        self.pos = []
        self.neg = []
        self.ratio = []
        self.loss = []
        self.mcc = []
        self.path = path


    def train(self):
        for round in range(1):
            try:
                os.mkdir('%s/%d' % (self.path, round))
            except FileExistsError:
                pass

            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            # set initial performance point
            model = self.model
            # print(torch.cuda.current_device())
            # print(torch.cuda.device_count(), torch.cuda.is_available())
            net = NN.NeuralNetwork(model=model,
                                    num_classes=2,
                                    batch_size=500,
                                    device_ids=[0],
                                    epochs=50)
            net.lr_fc = 0.0001

            net.initiate(self.dataset[label_ind.index], self.labels[label_ind.index])

            net.predict(self.testset)
            pred = net.preds
            weight = []

            conf_mat = confusion_matrix(y_true=self.testlab, y_pred=pred)
            precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
            recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
            specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
            gmeans = sqrt(recall * specificity)
            f1 = metrics.f1_score(y_true=self.testlab, y_pred=pred)
            auc = metrics.roc_auc_score(y_true=self.testlab, y_score=pred)
            accuracy = self.alibox.calc_performance_metric(y_true=self.testlab,
                                                           y_pred=pred.reshape(list(self.testlab.shape)),
                                                           performance_metric='accuracy_score')
            self.auc.append(auc)
            self.acc.append(accuracy)
            self.f1.append(f1)
            self.gmeans.append(gmeans)
            self.recall.append(recall)
            self.precision.append(precision)
            self.specificity.append(specificity)
            all = len(label_ind) + len(unlab_ind)
            lab_init = len(label_ind)
            lab = list(self.labels[label_ind.index])
            self.pos.append(lab.count(1))
            self.neg.append(lab.count(0))
            self.ratio.append(lab.count(0)/lab.count(1))
            tn, tp, fp, fn = conf_mat[0,0], conf_mat[1,1], conf_mat[0,1], conf_mat[1,0]
            mcc = ((tn*tp)-(fn*fp))/sqrt((tn+fp)*(tn+fn)*(tp+fp)*(tp+fn))
            self.mcc.append(mcc)

            saver.set_initial_point(gmeans)
            iteration = 0

            while not self.stopping_criterion.is_stop():
                # select subsets of Uind samples according to query strategy
                iteration += 1

                if self.phase == 'active':
                    net.predict(self.dataset[unlab_ind.index])
                    prob_pred = net.probablistic_matrix()

                    if len(label_ind) < all*0.3:
                        if iteration % 10:
                            select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                                  predict=prob_pred,
                                                                                  batch_size=int(lab_init*0.4))
                                                                                  # batch_size=1)
                        else:
                            select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init*0.4))
                            # select_ind = self.random.select(label_ind, unlab_ind, batch_size=1)
                    else:
                        select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                                  predict=prob_pred,
                                                                                  batch_size=int(len(label_ind)*0.4))
                                                                                  # batch_size=1)
                elif self.phase == 'passive':
                    if len(label_ind) < all*0.3:
                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init*0.4))
                        # select_ind = self.random.select(label_ind, unlab_ind, batch_size=1)
                    else:
                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(len(label_ind)*0.4))
                        # select_ind = self.random.select(label_ind, unlab_ind, batch_size=1)

                # print(select_ind)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                # update model and calc performance accoding to the updated model
                loss = net.train(self.dataset[label_ind.index], self.labels[label_ind.index])

                # if not iteration%2:
                net.predict(self.testset)
                pred = net.preds

                conf_mat = confusion_matrix(y_true=self.testlab, y_pred=pred)
                precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
                recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
                specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
                gmeans = sqrt(recall * specificity)
                f1 = metrics.f1_score(y_true=self.testlab, y_pred=pred)
                auc = metrics.roc_auc_score(y_true=self.testlab, y_score=pred)
                accuracy = self.alibox.calc_performance_metric(y_true=self.testlab,
                                                               y_pred=pred.reshape(list(self.testlab.shape)),
                                                               performance_metric='accuracy_score')
                self.auc.append(auc)
                self.acc.append(accuracy)
                self.f1.append(f1)
                self.gmeans.append(gmeans)
                self.recall.append(recall)
                self.precision.append(precision)
                self.specificity.append(specificity)
                lab = list(self.labels[label_ind.index])
                self.pos.append(lab.count(1))
                self.neg.append((lab.count(0)))
                self.ratio.append(lab.count(0)/lab.count(1))
                self.loss.append(loss)
                tn, tp, fp, fn = conf_mat[0,0], conf_mat[1,1], conf_mat[0,1], conf_mat[1,0]
                mcc = ((tn*tp)-(fn*fp))/sqrt((tn+fp)*(tn+fn)*(tp+fp)*(tp+fn))
                self.mcc.append(mcc)

                # save the results
                st = self.alibox.State(select_ind, gmeans)
                saver.add_state(st)
                saver.save()

                self.stopping_criterion.update_information(saver)
                lab = list(self.labels[label_ind.index])
                print('\n class \n0 and 1\n',lab.count(0), lab.count(1))
                print('\n',conf_mat)
                torch.save(self.model, './%s/%d/model%d' % (self.path, round, iteration))

            self.stopping_criterion.reset()
            self.unc_result.append(copy.deepcopy(saver))
            joblib.dump(self.auc, './%s/%d/auc' % (self.path, round))
            joblib.dump(self.acc, './%s/%d/acc' % (self.path, round))
            joblib.dump(self.f1, './%s/%d/f1' % (self.path, round))
            joblib.dump(self.gmeans, './%s/%d/gmeans' % (self.path, round))
            joblib.dump(self.recall, './%s/%d/recall' % (self.path, round))
            joblib.dump(self.precision, './%s/%d/precision' % (self.path, round))
            joblib.dump(self.specificity, './%s/%d/specificity' % (self.path, round))
            joblib.dump(self.pos, './%s/%d/pos' % (self.path, round))
            joblib.dump(self.neg, './%s/%d/neg' % (self.path, round))
            joblib.dump(self.ratio, './%s/%d/ratio' % (self.path, round))
            joblib.dump(self.mcc, './%s/%d/mcc' % (self.path, round))
        self.analyser = self.alibox.get_experiment_analyser(x_axis='num_of_queries')
        self.analyser.add_method(method_name='QueryInstanceUncertaity-lc', method_results=self.unc_result)
        print(self.analyser)
        # self.analyser.plot_learning_curves(title=self.title, std_area=True)


class TorchRegressionFold:
    def __init__(self, dataset, labels, testset, testlab, model, phase, path, stopping,
                 measure='nearest_neighbor', distance='linear'):
        self.dataset = dataset
        self.labels = labels
        self.testset = testset
        self.testlab = testlab
        self.model = model
        self.phase = phase
        self.classes = int(max(labels))
        self.alibox = ToolBox(X=dataset, y=np.asarray([0]* len(labels), dtype=np.int), query_type='AllLabels', saving_path='./%s' % path)
        self.alibox.split_AL(test_ratio=0, initial_label_rate=0.05, split_count=1)
        self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
        self.measure = measure
        if measure == 'residue': self.query_strategy = QueryInstanceResidueRegressor(X=self.dataset,
                                                                                     y=self.labels,
                                                                                     distance=distance)
        else: self.query_strategy = QueryInstanceDistribution(measure=measure)
        self.random = QueryRandom()
        self.unc_result = []
        self.title = ''
        self.loss = []
        self.path = path
        self.one = self.two = self.three = self.four = self.five = self.six = None
        self.max, self.mae, self.mse, self.evs, self.r2 = [], [], [], [], []
        self.sample = []


    def train(self):
        from sklearn.metrics import (mean_squared_log_error as msle,
                                     max_error as max,
                                     mean_absolute_error as mae,
                                     mean_squared_error as mse,
                                     explained_variance_score as evs,
                                     r2_score as r2,
                                     mean_tweedie_deviance as tweedie)
        for round in range(1):
            try:
                os.mkdir('%s/%d' % (self.path, round))
            except FileExistsError:
                pass

            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            # set initial performance point
            model = self.model
            net = NN.NeuralNetworkRegressor(model=model,
                                    batch_size=1,
                                    device_ids=[0],
                                    epochs=50)
            net.lr_fc = 0.01

            net.initiate(self.dataset[label_ind.index], self.labels[label_ind.index])

            net.predict(self.testset)
            pred = net.preds

            # evaluation
            all = len(label_ind) + len(unlab_ind)
            lab_init = len(label_ind)
            self.mse.append(mse(self.testlab, pred))
            self.mae.append(mae(self.testlab, pred))
            self.max.append(max(self.testlab, pred))
            self.evs.append(evs(self.testlab, pred))
            self.r2.append(r2(self.testlab, pred))
            self.sample.append(len(label_ind.index))

            saver.set_initial_point(mse(self.testlab, pred))
            iteration = 0

            while not self.stopping_criterion.is_stop():
                # select subsets of Uind samples according to query strategy
                iteration += 1

                lr_fc = net.lr_fc * (1 - len(label_ind.index)/(all*1.001))
                for p in net.optimizer.param_groups: p['lr'] = lr_fc
                print('learning rate is', net.optimizer.state_dict()['param_groups'][0]['lr'])

                if self.phase == 'active':
                    if self.measure != 'residue':
                        net.predict(self.dataset[unlab_ind.index])
                    else:
                        net.predict(self.dataset[label_ind])
                    pred = net.preds

                    if self.measure == 'distance':
                        if iteration == 1: self._update_previous_prediction(pred)
                        else: self._update_previous_prediction(pred, select_ind, unlab_ind_save)
                        previous = self._get_previous_prediction()
                    else: previous = None

                    if len(label_ind) < all*0.6:
                        if iteration % 10:
                            select_ind = self.query_strategy.select_by_prediction(unlabel_index=unlab_ind,
                                                                                  predict=pred,
                                                                                  labels=self.labels[label_ind.index],
                                                                                  batch_size=int(lab_init*1),
                                                                                  X_lab=self.dataset[label_ind.index],
                                                                                  X_unlab=self.dataset[unlab_ind.index],
                                                                                  previous=previous)
                        else:
                            select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init*1))
                    else:
                        select_ind = self.query_strategy.select_by_prediction(unlabel_index=unlab_ind,
                                                                              predict=pred,
                                                                              labels=self.labels[label_ind.index],
                                                                              batch_size=int(len(label_ind)*0.3),
                                                                              X_lab=self.dataset[label_ind.index],
                                                                              X_unlab=self.dataset[unlab_ind.index],
                                                                              previous=previous)
                elif self.phase == 'passive':
                    if len(label_ind) < all*0.6:
                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init*1))
                        # select_ind = self.random.select(label_ind, unlab_ind, batch_size=1)
                    else:
                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(len(label_ind)*0.3))
                        # select_ind = self.random.select(label_ind, unlab_ind, batch_size=1)

                # update the datasets and previous prediction
                unlab_ind_save = unlab_ind.index
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)


                # update model and calc performance accoding to the updated model
                loss = net.train(self.dataset[label_ind.index], self.labels[label_ind.index])

                # if not iteration%2:
                net.predict(self.testset)
                pred = net.preds

                # evaluation
                self.mse.append(mse(self.testlab, pred))
                self.mae.append(mae(self.testlab, pred))
                self.max.append(max(self.testlab, pred))
                self.evs.append(evs(self.testlab, pred))
                self.r2.append(r2(self.testlab, pred))
                self.sample.append(len(label_ind.index))
                self.loss.append(loss)

                # save the results
                st = self.alibox.State(select_ind, mse(self.testlab, pred))
                saver.add_state(st)
                saver.save()

                self.stopping_criterion.update_information(saver)
                torch.save(self.model, './%s/%d/model%d' % (self.path, round, iteration))

            self.stopping_criterion.reset()
            self.unc_result.append(copy.deepcopy(saver))
            joblib.dump(self.mse, './%s/%d/mse' % (self.path, round))
            joblib.dump(self.mae, './%s/%d/mae' % (self.path, round))
            joblib.dump(self.max, './%s/%d/max' % (self.path, round))
            joblib.dump(self.evs, './%s/%d/evs' % (self.path, round))
            joblib.dump(self.r2, './%s/%d/r2' % (self.path, round))
            joblib.dump(self.sample, './%s/%d/sample' % (self.path, round))
            joblib.dump(self.loss, './%s/%d/loss' % (self.path, round))
            joblib.dump(self.testlab, './%s/%d/testlab' % (self.path, round))
            joblib.dump(pred, './%s/%d/pred' % (self.path, round))
        self.analyser = self.alibox.get_experiment_analyser(x_axis='num_of_queries')
        self.analyser.add_method(method_name='QueryInstanceDistribution-distance', method_results=self.unc_result)
        print(self.analyser)

    def _update_previous_prediction(self, new, selected=None, unlab=None):
        if self.six is not None: del_ind = [unlab.index(i) for i in selected]
        if self.two is not None: self.one = np.delete(self.two, del_ind)
        if self.three is not None: self.two = np.delete(self.three, del_ind)
        if self.four is not None: self.three = np.delete(self.four, del_ind)
        if self.five is not None: self.four = np.delete(self.five, del_ind)
        if self.six is not None: self.five = np.delete(self.six, del_ind)
        self.six = new

    def _get_previous_prediction(self):
        if self.one is not None: return np.vstack((self.one, self.two, self.three,
                                       self.four, self.five, self.six))
        elif self.two is not None: return np.vstack((self.two, self.three,
                                       self.four, self.five, self.six))
        elif self.three is not None: return np.vstack((self.three, self.four, self.five, self.six))

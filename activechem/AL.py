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


class Random:
    def __init__(self, dataset, labels, test_ratio, model, path, stopping):
        self.dataset = dataset
        self.labels = labels
        self.classes = int(max(labels))
        self.test_ratio = test_ratio
        self.model = model
        self.alibox = ToolBox(X=dataset, y=labels, query_type='AllLabels', saving_path='./%s' % path)
        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=0.05, split_count=5)
        self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
        self.query_strategy = QueryInstanceUncertainty(X=dataset, y=labels, measure='margin')
        # self.query_strategy = QueryInstanceQBC(disagreement='KL_divergence')
        self.random = QueryRandom()
        self.unc_result = []
        self.title = ''
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.specificity = []
        self.sensitivity = []
        self.auc = []
        self.f1 = []
        self.path = path


    def classify(self):
        for round in range(5):
            os.mkdir('%s/%d' % (self.path, round))
            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            # set initial performance point
            model = self.model
            net = CNN.NeuralNetwork(model=model,
                                    num_classes=2,
                                    batch_size=500,
                                    device_ids=[0,1,2,3],
                                    epochs=10)
            net.lr_fc = 0.1
            net.initiate(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.predict(self.dataset[test_idx])
            pred = net.preds
            weight = []

            conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
            precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
            recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
            specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
            sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
            gmeans = sqrt(recall * specificity)
            f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
            auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
            accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                           y_pred=pred.reshape(list(self.labels[test_idx].shape)),
                                                           performance_metric='accuracy_score')
            self.auc.append(auc)
            self.acc.append(accuracy)
            self.f1.append(f1)
            self.gmeans.append(gmeans)
            self.recall.append(recall)
            self.precision.append(precision)
            self.specificity.append(specificity)
            self.pos = []
            self.neg = []
            self.loss = []
            self.mcc = []
            all = len(label_ind) + len(unlab_ind)
            lab_init = len(label_ind)

            saver.set_initial_point(gmeans)
            iteration = 0

            while not self.stopping_criterion.is_stop():
                # select subsets of Uind samples according to query strategy
                # net.predict(self.dataset[unlab_ind.index])
                # prob_pred = net.probablistic_matrix()
                # prob_pred = nn.predict_proba(self.dataset[unlab_ind.index])

                lab_ratio = len(label_ind) / all
                batch = int(len(label_ind)/3)
                iteration += 1
                if len(label_ind) < all*0.3:
                    select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init*0.4))
                else:
                    select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(len(label_ind)*0.4))

                # print(select_ind)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                # update model and calc performance accoding to the updated model
                loss = net.train(self.dataset[label_ind.index], self.labels[label_ind.index])
                # nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
                # if not iteration%2:
                net.predict(self.dataset[test_idx])
                pred = net.preds

                conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
                precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
                recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
                specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
                sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
                gmeans = sqrt(recall * specificity)
                f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
                auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
                accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                               y_pred=pred.reshape(list(self.labels[test_idx].shape)),
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
                self.loss.append(loss)
                tn = conf_mat[0,0]
                tp = conf_mat[1,1]
                fp = conf_mat[0,1]
                fn = conf_mat[1,0]
                mcc = ((tn*tp)-(fn*fp))/sqrt((tn+fp)*(tn+fn)*(tp+fp)*(tp+fn))
                self.mcc.append(mcc)


                # save the results
                st = self.alibox.State(select_ind, gmeans)
                saver.add_state(st)
                saver.save()

                self.stopping_criterion.update_information(saver)
                lab = list(self.labels[label_ind.index])
                print('\n class \n0 and 1\n',lab.count(0), lab.count(1))
                print('\n',confusion_matrix(y_true=self.labels[test_idx], y_pred=pred))
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
            joblib.dump(self.mcc, './%s/%d/mcc' % (self.path, round))
        self.analyser = self.alibox.get_experiment_analyser(x_axis='num_of_queries')
        self.analyser.add_method(method_name='Random', method_results=self.unc_result)
        print(self.analyser)
        # self.analyser.plot_learning_curves(title=self.title, std_area=True)


class Torch:
    def __init__(self, dataset, labels, test_ratio, model, path, stopping, nn=None):
        self.dataset = dataset
        self.labels = labels
        self.classes = int(max(labels))
        self.test_ratio = test_ratio
        self.model = model
        self.alibox = ToolBox(X=dataset, y=labels, query_type='AllLabels', saving_path='./%s' % path)
        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=0.05, split_count=5)
        self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
        self.query_strategy = QueryInstanceUncertainty(X=dataset, y=labels, measure='least_confident')
        # self.query_strategy = QueryInstanceQBC(disagreement='KL_divergence')
        self.random = QueryRandom()
        self.nn = nn
        self.unc_result = []
        self.title = ''
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.specificity = []
        self.sensitivity = []
        self.auc = []
        self.f1 = []
        self.path = path


    def classify(self):
        for round in range(5):
            os.mkdir('%s/%d' % (self.path, round))
            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            # set initial performance point
            model = self.model
            nn = self.nn
            net = CNN.NeuralNetwork(model=model,
                                    num_classes=2,
                                    batch_size=500,
                                    device_ids=[0,1,2,3],
                                    epochs=10)
            net.lr_fc = 0.1
            if nn:
                nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index], batch_size=300)
            net.initiate(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.predict(self.dataset[test_idx])
            pred = net.preds
            weight = []

            conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
            precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
            recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
            specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
            sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
            gmeans = sqrt(recall * specificity)
            f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
            auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
            accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                           y_pred=pred.reshape(list(self.labels[test_idx].shape)),
                                                           performance_metric='accuracy_score')
            self.auc.append(auc)
            self.acc.append(accuracy)
            self.f1.append(f1)
            self.gmeans.append(gmeans)
            self.recall.append(recall)
            self.precision.append(precision)
            self.specificity.append(specificity)
            self.pos = []
            self.neg = []
            self.loss = []
            self.mcc = []
            all = len(label_ind) + len(unlab_ind)
            lab_init = len(label_ind)

            saver.set_initial_point(gmeans)
            iteration = 0

            while not self.stopping_criterion.is_stop():
                # select subsets of Uind samples according to query strategy
                if nn:
                    prob_pred = nn.predict_proba(self.dataset[unlab_ind.index])
                else:
                    net.predict(self.dataset[unlab_ind.index])
                    prob_pred = net.probablistic_matrix()
                # print(prob_pred)
                # print(prob_pred.shape, len(unlab_ind.index))

                lab_ratio = len(label_ind) / all
                batch = int(len(label_ind)/3)
                iteration += 1
                if len(label_ind) < all*0.3:
                    if iteration % 10:
                        select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                              predict=prob_pred,
                                                                              batch_size=int(lab_init*0.4))
                    else:
                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init*0.4))
                else:
                    select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                              predict=prob_pred,
                                                                              batch_size=int(len(label_ind)*0.4))

                # print(select_ind)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                # update model and calc performance accoding to the updated model
                loss = net.train(self.dataset[label_ind.index], self.labels[label_ind.index])
                if nn:
                    nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
                # if not iteration%2:
                net.predict(self.dataset[test_idx])
                pred = net.preds

                conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
                precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
                recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
                specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
                # sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
                gmeans = sqrt(recall * specificity)
                f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
                auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
                accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                               y_pred=pred.reshape(list(self.labels[test_idx].shape)),
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
                self.loss.append(loss)
                tn = conf_mat[0,0]
                tp = conf_mat[1,1]
                fp = conf_mat[0,1]
                fn = conf_mat[1,0]
                mcc = ((tn*tp)-(fn*fp))/sqrt((tn+fp)*(tn+fn)*(tp+fp)*(tp+fn))
                self.mcc.append(mcc)

                # save the results
                st = self.alibox.State(select_ind, gmeans)
                saver.add_state(st)
                saver.save()

                self.stopping_criterion.update_information(saver)
                lab = list(self.labels[label_ind.index])
                print('\n class \n0 and 1\n',lab.count(0), lab.count(1))
                print('\n',confusion_matrix(y_true=self.labels[test_idx], y_pred=pred))
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
            joblib.dump(self.mcc, './%s/%d/mcc' % (self.path, round))
        self.analyser = self.alibox.get_experiment_analyser(x_axis='num_of_queries')
        self.analyser.add_method(method_name='QueryInstanceUncertaity-lc', method_results=self.unc_result)
        print(self.analyser)
        # self.analyser.plot_learning_curves(title=self.title, std_area=True)


class RandomSlow:
    def __init__(self, dataset, labels, test_ratio, model, path, stopping):
        self.dataset = dataset
        self.labels = labels
        self.classes = int(max(labels))
        self.test_ratio = test_ratio
        self.model = model
        self.alibox = ToolBox(X=dataset, y=labels, query_type='AllLabels', saving_path='./%s' % path)
        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=0.005, split_count=5)
        self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
        self.query_strategy = QueryInstanceUncertainty(X=dataset, y=labels, measure='margin')
        # self.query_strategy = QueryInstanceQBC(disagreement='KL_divergence')
        self.random = QueryRandom()
        self.unc_result = []
        self.title = ''
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.specificity = []
        self.sensitivity = []
        self.auc = []
        self.f1 = []
        self.path = path


    def classify(self):
        for round in range(1):
            os.mkdir('%s/%d' % (self.path, round))
            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            # set initial performance point
            model = self.model
            net = CNN.NeuralNetwork(model=model,
                                    num_classes=2,
                                    batch_size=500,
                                    device_ids=[0,1,2,3],
                                    epochs=10)
            net.lr_fc = 0.1
            net.initiate(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.predict(self.dataset[test_idx])
            pred = net.preds
            weight = []

            conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
            precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
            recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
            specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
            sensitivity = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
            gmeans = sqrt(recall * specificity)
            f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
            auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
            accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                           y_pred=pred.reshape(list(self.labels[test_idx].shape)),
                                                           performance_metric='accuracy_score')
            self.auc.append(auc)
            self.acc.append(accuracy)
            self.f1.append(f1)
            self.gmeans.append(gmeans)
            self.recall.append(recall)
            self.precision.append(precision)
            self.specificity.append(specificity)
            self.pos = []
            self.neg = []
            self.loss = []
            self.mcc = []
            all = len(label_ind) + len(unlab_ind)
            lab_init = len(label_ind)

            saver.set_initial_point(gmeans)
            iteration = 0

            while not self.stopping_criterion.is_stop():
                # select subsets of Uind samples according to query strategy
                # net.predict(self.dataset[unlab_ind.index])
                # prob_pred = net.probablistic_matrix()
                # prob_pred = nn.predict_proba(self.dataset[unlab_ind.index])

                lab_ratio = len(label_ind) / all
                batch = int(len(label_ind)/3)
                iteration += 1
                if len(label_ind) < all*0.3:
                    select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init))
                else:
                    select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(len(label_ind)*0.4))

                # print(select_ind)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                # update model and calc performance accoding to the updated model
                loss = net.train(self.dataset[label_ind.index], self.labels[label_ind.index])
                # nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
                # if not iteration%2:
                net.predict(self.dataset[test_idx])
                pred = net.preds

                conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
                precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
                recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
                specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
                sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
                gmeans = sqrt(recall * specificity)
                f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
                auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
                accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                               y_pred=pred.reshape(list(self.labels[test_idx].shape)),
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
                self.loss.append(loss)
                tn = conf_mat[0,0]
                tp = conf_mat[1,1]
                fp = conf_mat[0,1]
                fn = conf_mat[1,0]
                mcc = ((tn*tp)-(fn*fp))/sqrt((tn+fp)*(tn+fn)*(tp+fp)*(tp+fn))
                self.mcc.append(mcc)

                # save the results
                st = self.alibox.State(select_ind, gmeans)
                saver.add_state(st)
                saver.save()

                self.stopping_criterion.update_information(saver)
                lab = list(self.labels[label_ind.index])
                print('\n class \n0 and 1\n',lab.count(0), lab.count(1))
                print('\n',confusion_matrix(y_true=self.labels[test_idx], y_pred=pred))
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
            joblib.dump(self.mcc, './%s/%d/mcc' % (self.path, round))
        self.analyser = self.alibox.get_experiment_analyser(x_axis='num_of_queries')
        self.analyser.add_method(method_name='Random', method_results=self.unc_result)
        print(self.analyser)
        # self.analyser.plot_learning_curves(title=self.title, std_area=True)


class TorchSlow:
    def __init__(self, dataset, labels, test_ratio, model, path, stopping, nn=None):
        self.dataset = dataset
        self.labels = labels
        self.classes = int(max(labels))
        self.test_ratio = test_ratio
        self.model = model
        self.alibox = ToolBox(X=dataset, y=labels, query_type='AllLabels', saving_path='./%s' % path)
        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=0.005, split_count=5)
        self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
        self.query_strategy = QueryInstanceUncertainty(X=dataset, y=labels, measure='least_confident')
        # self.query_strategy = QueryInstanceQBC(disagreement='KL_divergence')
        self.random = QueryRandom()
        self.nn = nn
        self.unc_result = []
        self.title = ''
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.specificity = []
        self.sensitivity = []
        self.auc = []
        self.f1 = []
        self.path = path


    def classify(self):
        for round in range(1):
            os.mkdir('%s/%d' % (self.path, round))
            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            # set initial performance point
            model = self.model
            nn = self.nn
            net = CNN.NeuralNetwork(model=model,
                                    num_classes=2,
                                    batch_size=500,
                                    device_ids=[0,1,2,3],
                                    epochs=10)
            net.lr_fc = 0.1
            if nn:
                nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.initiate(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.predict(self.dataset[test_idx])
            pred = net.preds
            weight = []

            conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
            precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
            recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
            specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
            sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
            gmeans = sqrt(recall * specificity)
            f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
            auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
            accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                           y_pred=pred.reshape(list(self.labels[test_idx].shape)),
                                                           performance_metric='accuracy_score')
            self.auc.append(auc)
            self.acc.append(accuracy)
            self.f1.append(f1)
            self.gmeans.append(gmeans)
            self.recall.append(recall)
            self.precision.append(precision)
            self.sensitivity.append(sensitivity)
            self.pos = []
            self.neg = []
            self.loss = []
            self.mcc = []
            all = len(label_ind) + len(unlab_ind)
            lab_init = len(label_ind)

            saver.set_initial_point(gmeans)
            iteration = 0

            while not self.stopping_criterion.is_stop():
                # select subsets of Uind samples according to query strategy
                if nn:
                    prob_pred = nn.predict_proba(self.dataset[unlab_ind.index])
                else:
                    net.predict(self.dataset[unlab_ind.index])
                    prob_pred = net.probablistic_matrix()
                # print(prob_pred)
                # print(prob_pred.shape, len(unlab_ind.index))

                lab_ratio = len(label_ind) / all
                batch = int(len(label_ind)/3)
                iteration += 1
                if len(label_ind) < all*0.3:
                    if iteration % 10:
                        select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                              predict=prob_pred,
                                                                              batch_size=int(lab_init))
                    else:
                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init))
                else:
                    select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                              predict=prob_pred,
                                                                              batch_size=int(len(label_ind)*0.4))

                # print(select_ind)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                # update model and calc performance accoding to the updated model
                loss = net.train(self.dataset[label_ind.index], self.labels[label_ind.index])
                if nn:
                    nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
                # if not iteration%2:
                net.predict(self.dataset[test_idx])
                pred = net.preds

                conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
                precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
                recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
                specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
                sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
                gmeans = sqrt(recall * specificity)
                f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
                auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
                accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                               y_pred=pred.reshape(list(self.labels[test_idx].shape)),
                                                               performance_metric='accuracy_score')
                self.auc.append(auc)
                self.acc.append(accuracy)
                self.f1.append(f1)
                self.gmeans.append(gmeans)
                self.recall.append(recall)
                self.precision.append(precision)
                self.sensitivity.append(sensitivity)
                lab = list(self.labels[label_ind.index])
                self.pos.append(lab.count(1))
                self.neg.append((lab.count(0)))
                self.loss.append(loss)
                tn = conf_mat[0,0]
                tp = conf_mat[1,1]
                fp = conf_mat[0,1]
                fn = conf_mat[1,0]
                mcc = ((tn*tp)-(fn*fp))/sqrt((tn+fp)*(tn+fn)*(tp+fp)*(tp+fn))
                self.mcc.append(mcc)

                # save the results
                st = self.alibox.State(select_ind, gmeans)
                saver.add_state(st)
                saver.save()

                self.stopping_criterion.update_information(saver)
                lab = list(self.labels[label_ind.index])
                print('\n class \n0 and 1\n',lab.count(0), lab.count(1))
                print('\n',confusion_matrix(y_true=self.labels[test_idx], y_pred=pred))
                torch.save(self.model, './%s/%d/model%d' % (self.path, round, iteration))

            self.stopping_criterion.reset()
            self.unc_result.append(copy.deepcopy(saver))
            joblib.dump(self.auc, './%s/%d/auc' % (self.path, round))
            joblib.dump(self.acc, './%s/%d/acc' % (self.path, round))
            joblib.dump(self.f1, './%s/%d/f1' % (self.path, round))
            joblib.dump(self.gmeans, './%s/%d/gmeans' % (self.path, round))
            joblib.dump(self.recall, './%s/%d/recall' % (self.path, round))
            joblib.dump(self.precision, './%s/%d/precision' % (self.path, round))
            joblib.dump(self.sensitivity, './%s/%d/sensitivity' % (self.path, round))
            joblib.dump(self.pos, './%s/%d/pos' % (self.path, round))
            joblib.dump(self.neg, './%s/%d/neg' % (self.path, round))
            joblib.dump(self.mcc, './%s/%d/mcc' % (self.path, round))
        self.analyser = self.alibox.get_experiment_analyser(x_axis='num_of_queries')
        self.analyser.add_method(method_name='QueryInstanceUncertaity-lc', method_results=self.unc_result)
        print(self.analyser)
        # self.analyser.plot_learning_curves(title=self.title, std_area=True)


class TorchSlowSmall:
    def __init__(self, dataset, labels, test_ratio, model, path, stopping, nn=None):
        self.dataset = dataset
        self.labels = labels
        self.classes = int(max(labels))
        self.test_ratio = test_ratio
        self.model = model
        self.alibox = ToolBox(X=dataset, y=labels, query_type='AllLabels', saving_path='./%s' % path)
        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=0.1, split_count=5)
        self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
        self.query_strategy = QueryInstanceUncertainty(X=dataset, y=labels, measure='margin')
        # self.query_strategy = QueryInstanceQBC(disagreement='KL_divergence')
        self.random = QueryRandom()
        self.nn = nn
        self.unc_result = []
        self.title = ''
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.specificity = []
        self.sensitivity = []
        self.auc = []
        self.f1 = []
        self.path = path


    def classify(self):
        for round in range(5):
            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            # set initial performance point
            model = self.model
            nn = self.nn
            net = CNN.NeuralNetwork(model=model,
                                    num_classes=2,
                                    batch_size=500,
                                    device_ids=[0,1,2,3],
                                    epochs=10)
            net.lr_fc = 0.1
            net.epochs = 8
            if nn:
                nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.initiate(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.predict(self.dataset[test_idx])
            pred = net.preds
            weight = []

            conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
            precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
            recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
            specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
            sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
            gmeans = sqrt(recall * specificity)
            f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
            auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
            accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                           y_pred=pred.reshape(list(self.labels[test_idx].shape)),
                                                           performance_metric='accuracy_score')
            self.auc.append(auc)
            self.acc.append(accuracy)
            self.f1.append(f1)
            self.gmeans.append(gmeans)
            self.recall.append(recall)
            self.precision.append(precision)
            self.sensitivity.append(sensitivity)
            self.pos = []
            self.neg = []
            self.loss = []
            all = len(label_ind) + len(unlab_ind)
            lab_init = len(label_ind)

            saver.set_initial_point(gmeans)
            iteration = 0

            while not self.stopping_criterion.is_stop():
                # select subsets of Uind samples according to query strategy
                if nn:
                    prob_pred = nn.predict_proba(self.dataset[unlab_ind.index])
                else:
                    net.predict(self.dataset[unlab_ind.index])
                    prob_pred = net.probablistic_matrix()
                # print(prob_pred)
                # print(prob_pred.shape, len(unlab_ind.index))

                lab_ratio = len(label_ind) / all
                batch = int(len(label_ind)/3)
                iteration += 1
                if iteration % 10:
                    select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                          predict=prob_pred,
                                                                          batch_size=30)
                else:
                    select_ind = self.random.select(label_ind, unlab_ind, batch_size=30)


                # print(select_ind)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                # update model and calc performance accoding to the updated model
                loss = net.train(self.dataset[label_ind.index], self.labels[label_ind.index])
                if nn:
                    nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
                # if not iteration%2:
                net.predict(self.dataset[test_idx])
                pred = net.preds

                conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
                precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
                recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
                specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
                sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
                gmeans = sqrt(recall * specificity)
                f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
                auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
                accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                               y_pred=pred.reshape(list(self.labels[test_idx].shape)),
                                                               performance_metric='accuracy_score')
                self.auc.append(auc)
                self.acc.append(accuracy)
                self.f1.append(f1)
                self.gmeans.append(gmeans)
                self.recall.append(recall)
                self.precision.append(precision)
                self.sensitivity.append(sensitivity)
                lab = list(self.labels[label_ind.index])
                self.pos.append(lab.count(1))
                self.neg.append((lab.count(0)))
                self.loss.append(loss)

                # save the results
                st = self.alibox.State(select_ind, gmeans)
                saver.add_state(st)
                saver.save()

                self.stopping_criterion.update_information(saver)
                lab = list(self.labels[label_ind.index])
                print('\n class \n0 and 1\n',lab.count(0), lab.count(1))
                print('\n',confusion_matrix(y_true=self.labels[test_idx], y_pred=pred))
                torch.save(self.model, './%s/model%d' % (self.path, iteration))

            self.stopping_criterion.reset()
            self.unc_result.append(copy.deepcopy(saver))
        self.analyser = self.alibox.get_experiment_analyser(x_axis='num_of_queries')
        self.analyser.add_method(method_name='QueryInstanceUncertaity-lc', method_results=self.unc_result)
        print(self.analyser)
        # self.analyser.plot_learning_curves(title=self.title, std_area=True)
        joblib.dump(self.auc, './%s/auc' % self.path)
        joblib.dump(self.acc, './%s/acc' % self.path)
        joblib.dump(self.f1, './%s/f1' % self.path)
        joblib.dump(self.gmeans, './%s/gmeans' % self.path)
        joblib.dump(self.recall, './%s/recall' % self.path)
        joblib.dump(self.precision, './%s/precision' % self.path)
        joblib.dump(self.sensitivity, './%s/sensitivity' % self.path)
        joblib.dump(self.pos, './%s/pos' % self.path)
        joblib.dump(self.neg, './%s/neg' % self.path)


class Torch2D:
    def __init__(self, dataset, labels, test_ratio, model, path, stopping, nn=None):
        self.dataset = dataset
        self.labels = labels
        self.classes = int(max(labels))
        self.test_ratio = test_ratio
        self.model = model
        self.uselessset = np.zeros(list(dataset.shape[:2]), dtype=np.int8)
        self.alibox = ToolBox(X=self.uselessset, y=labels, query_type='AllLabels', saving_path='./%s' % path)
        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=0.05, split_count=5)
        self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
        self.query_strategy = QueryInstanceUncertainty(X=self.uselessset, y=labels, measure='least_confident')
        # self.query_strategy = QueryInstanceQBC(disagreement='KL_divergence')
        self.random = QueryRandom()
        self.nn = nn
        self.unc_result = []
        self.title = ''
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.specificity = []
        self.sensitivity = []
        self.auc = []
        self.f1 = []
        self.path = path


    def classify(self):
        for round in range(1):
            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            # set initial performance point
            model = self.model
            nn = self.nn
            net = CNN.NeuralNetwork(model=model,
                                    num_classes=2,
                                    batch_size=500,
                                    device_ids=[0,1,2,3],
                                    epochs=25)
            net.lr_fc = 0.1
            net.lr_conv = 0.1
            if nn:
                nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.initiate(self.dataset[label_ind.index], self.labels[label_ind.index])
            net.predict(self.dataset[test_idx])
            pred = net.preds
            weight = []

            conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
            precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
            recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
            specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
            sensitivity = conf_mat[0,0]/(conf_mat[1,1] + conf_mat[1,0])
            gmeans = sqrt(recall * specificity)
            f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
            auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
            accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                           y_pred=pred.reshape(list(self.labels[test_idx].shape)),
                                                           performance_metric='accuracy_score')
            self.auc.append(auc)
            self.acc.append(accuracy)
            self.f1.append(f1)
            self.gmeans.append(gmeans)
            self.recall.append(recall)
            self.precision.append(precision)
            self.specificity.append(specificity)
            self.pos = []
            self.neg = []
            self.loss = []
            all = len(label_ind) + len(unlab_ind)
            lab_init = len(label_ind)

            saver.set_initial_point(gmeans)
            iteration = 0

            while not self.stopping_criterion.is_stop():
                # select subsets of Uind samples according to query strategy
                if nn:
                    prob_pred = nn.predict_proba(self.dataset[unlab_ind.index])
                else:
                    net.predict(self.dataset[unlab_ind.index])
                    prob_pred = net.probablistic_matrix()
                # print(prob_pred)
                # print(prob_pred.shape, len(unlab_ind.index))

                lab_ratio = len(label_ind) / all
                batch = int(len(label_ind)/3)
                iteration += 1
                if len(label_ind) < all*0.3:
                    if iteration % 10:
                        select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                              predict=prob_pred,
                                                                              batch_size=int(lab_init*0.4))
                    else:
                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(lab_init*0.4))
                else:
                    select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                              predict=prob_pred,
                                                                              batch_size=int(len(label_ind)*0.4))

                # print(select_ind)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                # update model and calc performance accoding to the updated model
                loss = net.train(self.dataset[label_ind.index], self.labels[label_ind.index])
                if nn:
                    nn.fit(self.dataset[label_ind.index], self.labels[label_ind.index])
                # if not iteration%2:
                net.predict(self.dataset[test_idx])
                pred = net.preds

                conf_mat = confusion_matrix(y_true=self.labels[test_idx], y_pred=pred)
                precision = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])
                recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
                specificity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])
                sensitivity = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
                gmeans = sqrt(recall * specificity)
                f1 = metrics.f1_score(y_true=self.labels[test_idx], y_pred=pred)
                auc = metrics.roc_auc_score(y_true=self.labels[test_idx], y_score=pred)
                accuracy = self.alibox.calc_performance_metric(y_true=self.labels[test_idx],
                                                               y_pred=pred.reshape(list(self.labels[test_idx].shape)),
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
                self.loss.append(loss)

                # save the results
                st = self.alibox.State(select_ind, gmeans)
                saver.add_state(st)
                saver.save()

                self.stopping_criterion.update_information(saver)
                lab = list(self.labels[label_ind.index])
                print('\n class \n0 and 1\n',lab.count(0), lab.count(1))
                print('\n',confusion_matrix(y_true=self.labels[test_idx], y_pred=pred))
                torch.save(self.model, './%s/model%d' % (self.path, iteration))

            self.stopping_criterion.reset()
            self.unc_result.append(copy.deepcopy(saver))
        self.analyser = self.alibox.get_experiment_analyser(x_axis='num_of_queries')
        self.analyser.add_method(method_name='QueryInstanceUncertaity-lc', method_results=self.unc_result)
        print(self.analyser)
        # self.analyser.plot_learning_curves(title=self.title, std_area=True)
        joblib.dump(self.auc, './%s/auc' % self.path)
        joblib.dump(self.acc, './%s/acc' % self.path)
        joblib.dump(self.f1, './%s/f1' % self.path)
        joblib.dump(self.gmeans, './%s/gmeans' % self.path)
        joblib.dump(self.recall, './%s/recall' % self.path)
        joblib.dump(self.precision, './%s/precision' % self.path)
        joblib.dump(self.specificity, './%s/specificity' % self.path)
        joblib.dump(self.pos, './%s/pos' % self.path)
        joblib.dump(self.neg, './%s/neg' % self.path)


# def ecfp():
#     acdpath = '../all_druglikeness_data/acd_clear_sdf.sdf'
#     wdipath = '../all_druglikeness_data/wdi_all_sdf.sdf'
#
#     acd = preprocessing.read_sdf(acdpath)
#     wdi = preprocessing.read_sdf(wdipath)
#     dataset = []
#
#     from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect as morgan
#     for mol in acd:
#         mol0 = np.array(list(morgan(mol,0,512).ToBitString()), dtype=np.int8)
#         mol1 = np.array(list(morgan(mol,1,512).ToBitString()), dtype=np.int8)
#         mol2 = np.array(list(morgan(mol,2,512).ToBitString()), dtype=np.int8)
#         mol3 = np.array(list(morgan(mol,3,512).ToBitString()), dtype=np.int8)
#         mol4 = np.array(list(morgan(mol,4,512).ToBitString()), dtype=np.int8)
#         mol = mol0+mol1+mol2+mol3+mol4
#         dataset.append(list(mol))
#     for mol in wdi:
#         mol0 = np.array(list(morgan(mol,0,512).ToBitString()), dtype=np.int8)
#         mol1 = np.array(list(morgan(mol,1,512).ToBitString()), dtype=np.int8)
#         mol2 = np.array(list(morgan(mol,2,512).ToBitString()), dtype=np.int8)
#         mol3 = np.array(list(morgan(mol,3,512).ToBitString()), dtype=np.int8)
#         mol4 = np.array(list(morgan(mol,4,512).ToBitString()), dtype=np.int8)
#         mol = mol0+mol1+mol2+mol3+mol4
#         dataset.append(list(mol))
#     labels = [0]*len(acd) + [1]*len(wdi)
#     labels = np.array(labels)
#     dataset = np.array(dataset)
#
#     joblib.dump(dataset, './exptorch/ecfp_4_dataset')
#     joblib.dump(labels, './exptorch/ecfp_4_lab')
#
# from sklearn.model_selection import KFold
# import os
#
# def ds(data, label, path):
#     kf = KFold(n_splits=5, shuffle=True)
#     for k, (train,test) in enumerate(kf.split(data, label)):
#         model = CNN.FcRDKit()
#         net = CNN.NeuralNetwork(model, batch_size=500, epochs=50)
#         net.initiate(data[train], label[train], batch_size=500)
#         net.predict(data[test])
#         yte = label[test]
#         ypr = net.preds
#         conf = confusion_matrix(yte, ypr)
#         auc = metrics.roc_auc_score(yte, ypr)
#         torch.save(model, './%s/model' % path)
#         joblib.dump(auc, '%s/auc' % path)
#         joblib.dump(conf, '%s/confmat' % path)
#         joblib.dump(label[train], '%s/trainlab' % path)
#         break
#
#
# def us(data, label, path):
#     kf = KFold(n_splits=5, shuffle=True)
#     for k, (train,test) in enumerate(kf.split(data, label)):
#         model = CNN.FcRDKit()
#         net = CNN.NeuralNetwork(model, batch_size=500, epochs=50)
#         net.initiate(data[train], label[train], batch_size=500)
#         net.predict(data[test])
#         yte = label[test]
#         ypr = net.preds
#         conf = confusion_matrix(yte, ypr)
#         auc = metrics.roc_auc_score(yte, ypr)
#         torch.save(model, './%s/model' % path)
#         joblib.dump(auc, '%s/auc' % path)
#         joblib.dump(conf, '%s/confmat' % path)
#         joblib.dump(label[train], '%s/trainlab' % path)
#
#         break

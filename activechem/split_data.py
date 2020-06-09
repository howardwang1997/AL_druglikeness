# k-fold, splitting T/v/T t/t, random selecting subset
# requiring sklearn numpy

# import numpy
from random import randint
import numpy as np
import pandas as pd


def appendnum(proceeded, iteration, max):
    add = randint(0,max-1)
    if add not in proceeded:
        proceeded.append(add)
        iteration.append(add)
    else:
        proceeded, iteration = appendnum(proceeded, iteration, max)
    # print(add)
    return proceeded, iteration


def select(max, number):
    selected = []
    for i in range(number):
        selected, useless = appendnum(selected, [], max)
    selected.sort()
    return selected


def collect_the_rest(max, proceeded):
    last = []
    proceeded.sort()
    n = 0
    probe = 0
    while n <= proceeded[-1]:
        flag = proceeded[probe]
        # print(n,probe,proceeded,last,flag)
        if n == flag:
            n += 1
            probe += 1
        else:
            for i in range(n, flag):
                last.append(i)
            n = flag
    if n != max:
        for i in range(n,max):
            last.append(i)
    return last


class Kfold:
    def __init__(self, all, k=2):
        self.all = all
        self.k = k

    def fold(self):
        all = self.all
        k = self.k
        sets = []
        proceeded = []
        size = int(all/k)
        for i in range(k-1):
            iteration = []
            for j in range(size):
                proceeded, one_iter = appendnum(proceeded, iteration, all)
            iteration.sort()
            sets.append(iteration)
        last = collect_the_rest(all, proceeded)
        delete = all%k
        # print(all,max,size, delete, last, sets,proceeded)
        if delete:
            for i in range(delete):
                last.pop(randint(0,len(last)-i-1))
        sets.append(last)
        return sets


class RandomSampling:
    def __init__(self, max, portion='', number='', devide=''):
        self.portion = portion
        self.number = number
        self.devide = devide
        self.max = max
        if type(portion) is float:
            self.portion = portion
        elif type(number) is int:
            self.number = number
        elif type(devide) is int:
            self.devide = devide
        else:
            raise ValueError('no valid value given')

    def by_portion(self):
        number = int(self.max*self.portion)
        return select(max=self.max, number=number)

    def by_number(self):
        return select(max=self.max, number=self.number)

    def by_devide(self):
        number = int(self.max/self.devide)
        return select(max=self.max, number=number)


class TTSplit:
    def __init__(self, max, method, train=None, test=None):
        # method in ['portion', 'number', 'devide'], train set first, must seperate all data
        self.max = max
        self.method = method
        self.train = train
        self.test = test
        if train:
            self.calc = self.train = train
        elif test:
            self.calc = self.test = test

    def split(self):
        method = self.method
        if method == 'portion':
            number = int(self.max * self.calc)
        elif method == 'number':
            number = self.calc
        elif method == 'devide':
            number = int(self.max / self.calc)
        else:
            raise ValueError('no valid splitting method (portion or number or devide)')
        selected = select(max=self.max, number=number)
        opposite = collect_the_rest(max=self.max, proceeded=selected)
        if self.train:
            return selected, opposite
        elif self.test:
            return opposite, selected


class TVTSplit:
    def __init__(self, max, train, validation, test='', method='portion'):
        # method in ['portion', 'number', 'devide'], train set first, must seperate all data
        self.max = max
        self.train = train
        self.validation = validation

    def split(self):
        method = self.method
        if method == 'portion':
            num_train = int(max * self.train)
            num_validation = int(max * self.validation)
        elif method == 'number':
            num_train = self.train
            num_validation = self.validation
        else:
            raise ValueError('no valid splitting method (portion or number)')
        train = select(max=self.max, number=num_train)
        proceeded = list(tuple(train))
        validation = []
        for i in range(num_validation):
            proceeded, validation = appendnum(proceeded, validation, max)
        test = collect_the_rest(max=self.max, proceeded=proceeded)
        return train, validation, test


class MergeDatasets:
    def __init__(self):
        self.datasets = []
        self.labels = []

    def add_datasets(self, df):
        self.datasets.append(df)

    def add_labels(self, list):
        self.labels.append(list)

    def datasets_merged(self):
        return pd.concat(self.datasets, ignore_index=True)

    def labels_merged(self):
        labels = []
        for i in self.labels:
            labels += i
        return labels

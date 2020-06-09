import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

import os
import numpy as np
import math
import random

import split_data as split
import preprocessing


def initialization(m):
    if type(m)==nn.Linear:
        # m.bias.data.fill_(0.5046640270812254)
        nn.init.uniform_(m.weight,-0.05,0.05)
#
# class Cnn1dMold2(nn.Module):
#     def __init__(selfnum_classes=2):
#         super(Cnn1dMold2, self).__init__()
#
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=0, stride=1)
#         self.actv1 = nn.ReLU()
#         self.pool1 = nn.AvgPool1d(kernel_size=2)
#
#         self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=3, padding=0, stride=1)
#         self.actv2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=2)
#
#         self.conv3 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=4, padding=0, stride=1)
#         self.actv3 = nn.LeakyReLU(negative_slope=0.01)
#         self.pool3 = nn.MaxPool2d(kernel_size=3)
#
#         self.conv4 = nn.Conv1d(in_channels=16, out_channels=20, kernel_size=4, padding=0, stride=1)
#         self.actv4 = nn.LeakyReLU(negative_slope=0.001)
#
#         self.fc1 = nn.Linear(in_features=20*60, out_features=480, bias=True)
#         self.fc2 = nn.Linear(in_features=480, out_features=2, bias=True)
#
#     def forward(self, input):
#         in_size = len(input)
#
#         output = self.pool1(self.actv1(self.conv1(input)))
#         output = self.pool2(self.actv2(self.conv2(output)))
#         output = self.pool3(self.actv3(self.conv3(output)))
#         output = self.actv4(self.conv4(output))
#         output = self.fc2(self.fc1(output))
#         return output
#
#
# class Cnn1dMACCS(nn.Module):
#     def __init__(self, num_classes=2):
#         super(Cnn1dMACCS, self).__init__()
#
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=4, padding=0, stride=1)
#         self.actv1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=2)# 5 * 82
#
#         self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=3, padding=0, stride=1)
#         self.actv2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=2)# 10 * 40
#
#         self.conv3 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=5, padding=0, stride=1)
#         self.actv3 = nn.ReLU()
#         self.pool3 = nn.AvgPool1d(kernel_size=2)# 16 * 18
#
#         self.conv4 = nn.Conv1d(in_channels=16, out_channels=20, kernel_size=4, padding=0, stride=1)
#         # self.actv4 = nn.LeakyReLU(negative_slope=0.001)# 20 * 15
#
#         # self.drop = nn.Dropout(0.5)
#         # self.drop2 = nn.functional.dropout()
#         self.fc1 = nn.Linear(in_features=20*15, out_features=num_classes, bias=True)
#         # self.fc2 = nn.Linear(in_features=128, out_features=num_classes, bias=True)
#
#     def forward(self, input):
#         in_size = len(input)
#         output = self.pool1(self.actv1(self.conv1(input)))
#         output = self.pool2(self.actv2(self.conv2(output)))
#         output = self.pool3(self.actv3(self.conv3(output)))
#         # output = self.actv4(self.conv4(output))
#         output = self.conv4(output)
#         # print(output.shape)
#         # output = self.drop(output)
#         output = output.view(in_size, -1)
#         # print(output.shape)
#         output = nn.functional.dropout(output, p=0.5, training=self.training)
#         # print(output.shape)
#         # print(output.shape)
#         # output = self.fc2(self.fc1(output))
#         output = self.fc1(output)
#         return output


class Cnn1dRDKit(nn.Module):
    def __init__(self, num_classes=2):
        super(Cnn1dRDKit, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=0, stride=1)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)# 5 * 99

        self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=4, padding=0, stride=1)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)# 10 * 48

        self.conv3 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=5, padding=0, stride=1)
        self.actv3 = nn.ReLU()
        self.pool3 = nn.AvgPool1d(kernel_size=2)# 16 * 22

        # self.conv4 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=3, padding=0, stride=1)
        # self.actv4 = nn.LeakyReLU(negative_slope=0.001)# 24 * 20

        # self.drop = nn.Dropout(0.5)
        # self.drop2 = nn.functional.dropout()
        self.fc1 = nn.Linear(in_features=16*22, out_features=200, bias=True)
        self.fc2 = nn.Linear(in_features=200, out_features=num_classes, bias=True)

    def forward(self, input):
        in_size = len(input)
        output = self.pool1(self.actv1(self.conv1(input)))
        output = self.pool2(self.actv2(self.conv2(output)))
        output = self.pool3(self.actv3(self.conv3(output)))
        # output = self.actv4(self.conv4(output))
        # output = self.conv4(output)
        # print(output.shape)
        # output = self.drop(output)
        output = output.view(in_size, -1)
        # print(output.shape)
        output = nn.functional.dropout(output, training=self.training)
        # print(output.shape)
        # print(output.shape)
        output = self.fc2(self.fc1(output))
        # output = self.fc1(output)
        return output


class FcRDKit(nn.Module):
    def __init__(self, num_classes=2):
        super(FcRDKit, self).__init__()

        self.fc1 = nn.Linear(in_features=200, out_features=300, bias=True)
        self.actv1 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=300, out_features=num_classes, bias=True)
        self.soft = nn.Softmax()

    def forward(self, input):
        in_size = len(input)
        output = input.view(in_size, -1)
        # print(output.shape)
        # output = nn.functional.dropout(output, p=0.5, training=self.training)
        # print(output.shape)
        # print(output.shape)

        # output = self.fc2(self.fc1(output))
        output = self.fc1(output)
        output = self.actv1(output)
        output = self.fc2(output)
        # output = self.soft(output)
        return output


class FcECFP(nn.Module):
    def __init__(self, num_classes=2):
        super(FcECFP, self).__init__()

        self.fc1 = nn.Linear(in_features=512, out_features=300, bias=True)
        self.actv1 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=300, out_features=num_classes, bias=True)

    def forward(self, input):
        in_size = len(input)
        output = input.view(in_size, -1)
        # print(output.shape)
        # output = nn.functional.dropout(output, p=0.5, training=self.training)
        # print(output.shape)
        # print(output.shape)

        # output = self.fc2(self.fc1(output))
        output = self.fc1(output)
        output = self.actv1(output)
        output = self.fc2(output)
        return output


class Cnn2dECFP(nn.Module):
    # bitstring = 512
    def __init__(self, num_classes=2):
        super(Cnn2dECFP,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=0)
        self.actv1 = nn.LeakyReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, padding=0)# 5 * 255

        self.conv2 = nn.Conv1d(in_channels=5, out_channels=12, kernel_size=4, stride=1, padding=0)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, padding=0)# 12 * 126

        self.conv3 = nn.Conv1d(in_channels=12, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.actv3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, padding=1)# 20 * 61

        self.conv4 = nn.Conv1d(in_channels=20, out_channels=30, kernel_size=5, stride=1, padding=0)
        self.actv4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=3, padding=0)# 30 * 19

        self.conv5 = nn.Conv1d(in_channels=30, out_channels=40, kernel_size=4, stride=1, padding=0)

        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=int(40*16/2), out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes, bias=True)

    def forward(self, input):
        output = self.pool1(self.actv1(self.conv1(input)))
        output = self.pool2(self.actv2(self.conv2(output)))
        output = self.pool3(self.actv3(self.conv3(output)))
        output = self.pool4(self.actv4(self.conv4(output)))
        output = self.conv5(output)
        output = output.view(in_size, -1)
        output = self.fc2(self.fc1(self.drop(output)))
        return output


class Cnn2dOH(nn.Module):
    def __init__(self, num_classes=2):
        super(Cnn2dOH,self).__init__()
        # 1 * 40 * 120
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=0)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)# 5 * 18 * 58

        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)# 10 * 7 * 27

        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4, stride=1, padding=0)
        self.actv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=0)# 20* 2 * 12

        self.conv4 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=2, stride=1, padding=0)
        self.actv4 = nn.ReLU() # 40 * 11

        self.drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=40*11, out_features=256, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes, bias=True)
        self.soft = nn.Softmax()

    def forward(self, input):
        in_size = len(input)
        # print(in_size)
        # print(input.shape,'in')
        output = self.pool1(self.actv1(self.conv1(input)))
        # print(output.shape,'1')
        output = self.pool2(self.actv2(self.conv2(output)))
        # print(output.shape,'2')
        output = self.pool3(self.actv3(self.conv3(output)))
        # print(output.shape,'3')
        output = self.actv4(self.conv4(output))
        output = output.view(in_size, -1)
        # print(output.shape,'4')
        output = self.fc2(self.fc1(self.drop(output)))
        output = self.soft(output)
        return output
#

def _get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()


def _print_params(model):
    blank = ' '
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key)<=15: key = key + (15-len(key))*blank
        w_variable_blank = ''
        if len(w_variable.shape) == 1:
            if w_variable.shape[0] >= 100: w_variable_blank = 8*blank
            else: w_variable_blank = 9*blank
        elif len(w_variable.shape) == 2:
            if w_variable.shape[0] >= 100: w_variable_blank = 2*blank
            else: w_variable_blank = 3*blank
        print(key,' ', w_variable.shape, ' ', w_variable_blank)


def _label_matrix(labels, num_classes=2):
    matrix = torch.zeros([len(labels), num_classes], dtype=torch.int8)
    for i in range(len(labels)):
        matrix[i][labels[i]] += 1
    return matrix.type(torch.LongTensor)


class AverageRecorder(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count


class NeuralNetwork:
    """
    for linear inputs, Fc or Cnn1d
    nn(model=None, num_classes=2, device_ids=[0], batch_size=10, epochs=5, model_filename='model_cnn', trained=False)
    nn.initiate(data_init, label_init)
    nn.train(data, labels) << data and labels are the whole labeled set, not the adding samples
    nn.predict(data) >> nn.outputs is [n_test, n_classes] like 2d ndarray of prediction,
                        nn.preds is [n_test] like 1d ndarray of prediction
    nn.probablistic_matrix() >> [n_test, n_classes] like 2d ndarray
    """
    def __init__(self, model=None,
                 num_classes=2,
                 device_ids=[0],
                 batch_size=10,
                 epochs=5,
                 model_filename='model_cnn',
                 trained=False):
        # please specify dtype in tensors data_init and label_init
        self.model = model
        self.epochs = epochs
        self.filename = model_filename
        self.lr_fc = 1e-3
        self.lr_conv = 1e-3
        self.cudaFlag = torch.cuda.is_available()
        self.device = device_ids
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.trained = trained

        if self.trained:
            self.model = torch.load(model_filename)

        if self.cudaFlag:
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            # self.model = self.model.cuda(0)

    def initiate(self, data_init, label_init, batch_size=10):
        self.batch_size = batch_size
        # self.epochs = epochs
        self.dataset = data_init
        self.labels = label_init
        self.num_data = len(self.labels)
        self.loss_list = []
        self.accuracy_list = []
        if self.num_classes == 2:
            self.positive = len(torch.nonzero(self.labels, out=None))
            self.weight_pos = 1 - self.positive/self.num_data


        self.acc = AverageRecorder()
        self.fc_param = self.model.parameters(self.model)
        l = list(map(id, self.model.parameters(self.model)))
        self.conv_param = (parameter for parameter in self.model.parameters() if id(parameter) not in l)
        # print(self.conv_param)
        # self.optimizer = optim.SGD([{'params': self.fc_param, 'lr': self.lr_fc}])
        self.optimizer = optim.SGD([{'params': self.fc_param, 'lr': self.lr_fc},
                                {'params': self.conv_param, 'lr': self.lr_conv}])
        self.weight = torch.tensor([1-self.weight_pos, self.weight_pos]).cuda(torch.cuda.current_device())
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)

        initset = TensorDataset(self.dataset, self.labels)
        trainloader = DataLoader(dataset=initset,
                                 batch_size=self.batch_size,
                                 shuffle=True)
        self.model.apply(initialization)
        # print(self.dataset.shape, self.labels.shape)
        for epoch in range(self.epochs*3):
            running_loss = 0.0
            print('epoch',epoch+1)
            for i, data in enumerate(trainloader):
                inputs, labels = data
                self.optimizer.zero_grad()
                if self.cudaFlag:
                    # print(torch.cuda.current_device())
                    inputs = inputs.cuda(torch.cuda.current_device())
                    labels = labels.cuda(torch.cuda.current_device())
                if self.batch_size == 1:
                    inputs, labels = Variable(torch.unsqueeze(inputs, dim=0).float(), requires_grad=True), \
                                     Variable(labels.long())
                else:
                    labels = Variable(labels.long())
                    inlist = None
                    for j in inputs:
                        if inlist is None:
                            inlist = torch.unsqueeze(torch.unsqueeze(j, dim=0), dim=0)
                        else:
                            inlist = torch.cat((inlist,torch.unsqueeze(torch.unsqueeze(j, dim=0), dim=0)),0)
                    # print(inputs)
                    # print(torch.tensor(inlist))
                    inputs = Variable(inlist.float(), requires_grad=True)
                # print(inputs)

                # print(inputs.shape)
                # print(inputs)
                outputs = self.model(inputs)

                # print(outputs, labels)
                # print(labels)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # print(loss.item())

                accuracy = _get_accuracy(outputs, labels, self.batch_size)
                self.acc.update(accuracy)

                running_loss += loss.item()

                self.trained = True
            print('initiate epoch loss :', running_loss/(i+1))

    def train(self, data, labels):
        # labels = torch.unsqueeze(labels, dim=0)
        # print(data.shape, labels.shape)
        # print(labels)
        if len(data.shape) == 1:
            data, labels = torch.unsqueeze(data, dim=0), torch.unsqueeze(labels, dim=0)

        trainset = TensorDataset(data, labels)
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=self.batch_size,
                                 shuffle=True)
        # print(data.shape)

        for epoch in range(self.epochs):
            running_loss = 0.0
            print('epoch', epoch+1)
            for i, data in enumerate(trainloader):
                inputs, labels = data
                # print(labels)
                if self.cudaFlag:
                    inputs = inputs.cuda(device=torch.cuda.current_device())
                    labels = labels.cuda(device=torch.cuda.current_device())
                # print(inputs.shape)
                if self.batch_size == 1:
                    inputs, labels = Variable(torch.unsqueeze(inputs, dim=0).float(), requires_grad=True),\
                                     Variable(labels.long())
                else:
                    labels = Variable(labels.long())
                    inlist = None
                    for j in inputs:
                        if inlist is None:
                            inlist = torch.unsqueeze(torch.unsqueeze(j, dim=0), dim=0)
                        else:
                            inlist = torch.cat((inlist,torch.unsqueeze(torch.unsqueeze(j, dim=0), dim=0)),0)
                    # print(inputs)
                    # print(torch.tensor(inlist))
                    inputs = Variable(inlist.float(), requires_grad=True)

                self.optimizer.zero_grad()
                self.model.train()

                outputs = self.model(inputs)

                # print(outputs.grad, inputs.grad)
                # print(labels, outputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print(inputs.grad)
                # print(loss.item())
                running_loss += loss.item()
                # for param in self.model.parameters():
                #     print(param)
                # print(outputs.grad, inputs.grad)
                # input()

                self.num_data += self.batch_size
                self.positive += len(torch.nonzero(labels, out=None))
                self.weight_pos = 1 - self.positive/self.num_data
                # self.weight = torch.tensor([1-self.weight_pos, self.weight_pos]).cuda(torch.cuda.current_device())
                # self.criterion = nn.CrossEntropyLoss(weight=self.weight)
                # print(self.weight, self.weight_pos, self.positive)
            print('train epoch loss:', running_loss/(i+1))
        return running_loss/(i+1)

    def predict(self, data):
        # testset = TensorDataset(data)
        if len(data.shape) == 1:
            data = torch.unsqueeze(data, dim=0)
        testloader = DataLoader(torch.tensor(data),
                                batch_size=self.batch_size,
                                shuffle=False)
        loss = AverageRecorder()

        # model = self.model.to(self.device)
        self.model.eval()
        outputs = []
        preds = []
        with torch.no_grad():
            for i,data in enumerate(testloader):
                if self.cudaFlag:
                    inputs = data
                    if self.cudaFlag:
                        inputs = inputs.cuda(device=torch.cuda.current_device())
                        # labels = labels.cuda(device=torch.cuda.current_device())
                if self.batch_size == 1:
                    inputs = Variable(torch.unsqueeze(inputs, dim=0).float(), requires_grad=True)
                else:
                    inlist = None
                    for i in inputs:
                        if inlist is None:
                            inlist = torch.unsqueeze(torch.unsqueeze(i, dim=0), dim=0)
                        else:
                            inlist = torch.cat((inlist,torch.unsqueeze(torch.unsqueeze(i, dim=0), dim=0)),0)
                    inputs = Variable(inlist.float(), requires_grad=True)
                output = self.model(inputs)
                _, pred = torch.max(output, 1)
                # print(output, _, pred)
                for i in range(len(pred)):
                    # outputs.append(torch.squeeze(output[i].cpu()))
                    # preds.append(np.array(torch.squeeze(pred[i].cpu())))
                    outputs.append(list(output[i].cpu()))
                    preds.append(int(pred[i].cpu()))

            # self.outputs = np.array([np.array(i) for i in outputs])
            self.outputs = np.array(outputs)
            self.preds = np.array(preds)
            # self.preds = preds

    def probablistic_matrix(self):
        from scipy.special import softmax, expit
        proba = self.outputs**3
        proba_pred = np.zeros(self.outputs.shape)
        for i in range(len(self.outputs)):
            proba_pred[i] = softmax(proba[i])
        return proba_pred


# class Network2dCNNcache:
#     def __init__(self, model,
#                  num_classes=2,
#                  device_ids=[0],
#                  epochs=5,
#                  model_filename='model_cnn',
#                  trained=False):
#         # please specify dtype in tensors data_init and label_init
#         self.model = model
#         self.epochs = epochs
#         self.filename = model_filename
#         self.lr_fc = 2e-3
#         self.lr_conv = 2e-3
#         self.cudaFlag = torch.cuda.is_available()
#         self.device = device_ids
#         self.num_classes = num_classes
#
#         self.trained = trained
#
#         if self.trained:
#             self.model = torch.load(model_filename)
#
#         if self.cudaFlag:
#             self.model = self.model.cuda()
#             self.model = nn.DataParallel(self.model, device_ids=device_ids)
#             # self.model = self.model.cuda(0)
#
#     def initiate(self, data_init, label_init, batch_size=1):
#         self.batch_size = batch_size
#         self.dataset = data_init
#         self.labels = label_init
#         self.num_data = len(self.labels)
#         self.loss_list = []
#         self.accuracy_list = []
#         if self.num_classes == 2:
#             self.positive = len(torch.nonzero(self.labels, out=None))
#             self.weight_pos = 1 - self.positive/self.num_data
#
#         self.acc = AverageRecorder()
#         self.fc_param = self.model.parameters(self.model)
#         l = list(map(id, self.model.parameters(self.model)))
#         self.conv_param = (parameter for parameter in self.model.parameters() if id(parameter) not in l)
#         self.optimizer = optim.SGD([{'params': self.fc_param, 'lr': self.lr_fc},
#                                 {'params': self.conv_param, 'lr': self.lr_conv}])
#
#         self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1-self.weight_pos,
#                                                                               self.weight_pos]).cuda(torch.cuda.current_device()))
#
#         initset = TensorDataset(self.dataset, self.labels)
#         trainloader = DataLoader(dataset=initset,
#                                  batch_size=self.batch_size,
#                                  shuffle=True)
#         running_loss = 0.0
#         # print(self.dataset.shape, self.labels.shape)
#         for i, data in enumerate(trainloader):
#             inputs, labels = data
#             self.optimizer.zero_grad()
#             if self.cudaFlag:
#                 # print(torch.cuda.current_device())
#                 inputs = inputs.cuda(torch.cuda.current_device())
#                 labels = labels.cuda(torch.cuda.current_device())
#             if self.batch_size == 1:
#                 inputs, labels = Variable(inputs, requires_grad=True), \
#                                  Variable(labels.long())
#             else:
#                 labels = Variable(labels.long())
#                 inlist = None
#                 for i in inputs:
#                     # NOT COMPLETED, Batch_size should be 1
#                     if inlist is None:
#                         inlist = torch.unsqueeze(torch.unsqueeze(i, dim=0), dim=0)
#                     else:
#                         inlist = torch.cat((inlist,torch.unsqueeze(torch.unsqueeze(i, dim=0), dim=0)),0)
#
#                 inputs = Variable(inlist.float(), requires_grad=True)
#
#             outputs = self.model(inputs)
#
#             loss = self.criterion(outputs, labels)
#             loss.backward()
#             self.optimizer.step()
#
#             accuracy = _get_accuracy(outputs, labels, self.batch_size)
#             self.acc.update(accuracy)
#
#             running_loss += loss.item()
#
#             self.trained = True
#         print('initiate loss :', running_loss)
#
#     def train(self, data, labels):
#         # labels = torch.unsqueeze(labels, dim=0)
#         # print(data.shape, labels.shape)
#         # print(labels)
#         if len(data.shape) == 1:
#             data, labels = torch.unsqueeze(data, dim=0), torch.unsqueeze(labels, dim=0)
#
#         trainset = TensorDataset(data, labels)
#         trainloader = DataLoader(dataset=trainset,
#                                  batch_size=self.batch_size,
#                                  shuffle=False)
#
#         for i, data in enumerate(trainloader):
#             inputs, labels = data
#             if self.cudaFlag:
#                 inputs = inputs.cuda(device=torch.cuda.current_device())
#                 labels = labels.cuda(device=torch.cuda.current_device())
#             inputs, labels = Variable(inputs, requires_grad=True),\
#                              Variable(labels.long())
#
#             self.optimizer.zero_grad()
#             self.model.train()
#
#             outputs = self.model(inputs)
#
#             # print(outputs.grad, inputs.grad)
#             print(labels, outputs)
#
#             loss = self.criterion(outputs, labels)
#             loss.backward()
#             self.optimizer.step()
#
#             print(loss.item())
#             # for param in self.model.parameters():
#             #     print(param)
#             # print(outputs.grad, inputs.grad)
#
#             self.num_data += 2
#             self.positive += len(torch.nonzero(labels, out=None))
#             self.weight_pos = 1 - self.positive/self.num_data
#             # self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1-self.weight_pos,
#             #                                                                       self.weight_pos]).cuda(torch.cuda.current_device()))
#
#     def predict(self, data):
#         # testset = TensorDataset(data)
#         if len(data.shape) == 1:
#             data = torch.unsqueeze(data, dim=0)
#         testloader = DataLoader(torch.tensor(data),
#                                 batch_size=self.batch_size,
#                                 shuffle=False)
#         loss = AverageRecorder()
#
#         # model = self.model.to(self.device)
#         self.model.eval()
#         outputs = []
#         preds = []
#         with torch.no_grad():
#             for i,data in enumerate(testloader):
#                 if self.cudaFlag:
#                     inputs = data
#                     if self.cudaFlag:
#                         inputs = inputs.cuda(device=torch.cuda.current_device())
#                         # labels = labels.cuda(device=torch.cuda.current_device())
#                 inputs = Variable(inputs, requires_grad=True)
#                 output = self.model(inputs)
#                 _, pred = torch.max(output, 1)
#                 # print(output, _, pred)
#                 outputs.append(torch.squeeze(output.cpu()))
#                 preds.append(torch.squeeze(pred.cpu()))
#
#             self.outputs = np.array([np.array(i) for i in outputs])
#             self.preds = np.array([np.array(i) for i in preds])
#
#
#     def probablistic_matrix(self):
#         from scipy import special
#         proba_pred =special.expit(self.outputs)
#         return proba_pred


# class Network2dCNN:
#     """
#     for Cnn2d
#     nn(model=None, num_classes=2, device_ids=[0], batch_size=10, epochs=5, model_filename='model_cnn', trained=False)
#     nn.initiate(data_init, label_init)
#     nn.train(data, labels) << data and labels are the whole labeled set, not the adding samples
#     nn.predict(data) >> nn.outputs is [n_test, n_classes] like 2d ndarray of prediction,
#                         nn.preds is [n_test] like 1d ndarray of prediction
#     nn.probablistic_matrix() >> [n_test, n_classes] like 2d ndarray
#     """
#     def __init__(self, model=None,
#                  num_classes=2,
#                  device_ids=[0],
#                  batch_size=10,
#                  epochs=5,
#                  model_filename='model_cnn',
#                  trained=False):
#         # please specify dtype in tensors data_init and label_init
#         self.model = model
#         self.epochs = epochs
#         self.filename = model_filename
#         self.lr_fc = 1e-3
#         self.lr_conv = 1e-3
#         self.cudaFlag = torch.cuda.is_available()
#         self.device = device_ids
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.trained = trained
#
#         if self.trained:
#             self.model = torch.load(model_filename)
#
#         if self.cudaFlag:
#             self.model = self.model.cuda()
#             self.model = nn.DataParallel(self.model, device_ids=device_ids)
#             # self.model = self.model.cuda(0)
#
#     def initiate(self, data_init, label_init, batch_size=10):
#         self.batch_size = batch_size
#         # self.epochs = epochs
#         self.dataset = data_init
#         self.labels = label_init
#         self.num_data = len(self.labels)
#         self.loss_list = []
#         self.accuracy_list = []
#         if self.num_classes == 2:
#             self.positive = len(torch.nonzero(self.labels, out=None))
#             self.weight_pos = 1 - self.positive/self.num_data
#
#
#         self.acc = AverageRecorder()
#         self.fc_param = self.model.parameters(self.model)
#         l = list(map(id, self.model.parameters(self.model)))
#         self.conv_param = (parameter for parameter in self.model.parameters() if id(parameter) not in l)
#         # print(self.conv_param)
#         # self.optimizer = optim.SGD([{'params': self.fc_param, 'lr': self.lr_fc}])
#         self.optimizer = optim.SGD([{'params': self.fc_param, 'lr': self.lr_fc},
#                                 {'params': self.conv_param, 'lr': self.lr_conv}])
#         self.weight = torch.tensor([1-self.weight_pos, self.weight_pos]).cuda(torch.cuda.current_device())
#         # self.criterion = nn.CrossEntropyLoss()
#         self.criterion = nn.CrossEntropyLoss(weight=self.weight)
#
#         initset = TensorDataset(self.dataset, self.labels)
#         trainloader = DataLoader(dataset=initset,
#                                  batch_size=self.batch_size,
#                                  shuffle=True)
#         self.model.apply(initialization)
#         # print(self.dataset.shape, self.labels.shape)
#         for epoch in range(self.epochs*3):
#             running_loss = 0.0
#             print('epoch',epoch+1)
#             for i, data in enumerate(trainloader):
#                 inputs, labels = data
#                 self.optimizer.zero_grad()
#                 if self.cudaFlag:
#                     # print(torch.cuda.current_device())
#                     inputs = inputs.cuda(torch.cuda.current_device())
#                     labels = labels.cuda(torch.cuda.current_device())
#                 if self.batch_size == 1:
#                     inputs, labels = Variable(torch.unsqueeze(inputs, dim=0).float(), requires_grad=True), \
#                                      Variable(labels.long())
#                 else:
#                     labels = Variable(labels.long())
#                     inlist = None
#                     for j in inputs:
#                         if inlist is None:
#                             inlist = torch.unsqueeze(torch.unsqueeze(j, dim=0), dim=0)
#                         else:
#                             inlist = torch.cat((inlist,torch.unsqueeze(torch.unsqueeze(j, dim=0), dim=0)),0)
#                     # print(inputs)
#                     # print(torch.tensor(inlist))
#                     inputs = Variable(inlist.float(), requires_grad=True)
#                 # print(inputs)
#
#                 # print(inputs.shape)
#                 # print(inputs)
#                 outputs = self.model(inputs)
#
#                 # print(outputs, labels)
#                 # print(labels)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#                 # print(loss.item())
#
#                 accuracy = _get_accuracy(outputs, labels, self.batch_size)
#                 self.acc.update(accuracy)
#
#                 running_loss += loss.item()
#
#                 self.trained = True
#             print('initiate epoch loss :', running_loss/(i+1))
#
#     def train(self, data, labels):
#         # labels = torch.unsqueeze(labels, dim=0)
#         # print(data.shape, labels.shape)
#         # print(labels)
#         if len(data.shape) == 1:
#             data, labels = torch.unsqueeze(data, dim=0), torch.unsqueeze(labels, dim=0)
#
#         trainset = TensorDataset(data, labels)
#         trainloader = DataLoader(dataset=trainset,
#                                  batch_size=self.batch_size,
#                                  shuffle=True)
#         # print(data.shape)
#
#         for epoch in range(self.epochs):
#             running_loss = 0.0
#             print('epoch', epoch+1)
#             for i, data in enumerate(trainloader):
#                 inputs, labels = data
#                 # print(labels)
#                 if self.cudaFlag:
#                     inputs = inputs.cuda(device=torch.cuda.current_device())
#                     labels = labels.cuda(device=torch.cuda.current_device())
#                 # print(inputs.shape)
#                 if self.batch_size == 1:
#                     inputs, labels = Variable(torch.unsqueeze(inputs, dim=0).float(), requires_grad=True),\
#                                      Variable(labels.long())
#                 else:
#                     labels = Variable(labels.long())
#                     inlist = None
#                     for j in inputs:
#                         if inlist is None:
#                             inlist = torch.unsqueeze(torch.unsqueeze(j, dim=0), dim=0)
#                         else:
#                             inlist = torch.cat((inlist,torch.unsqueeze(torch.unsqueeze(j, dim=0), dim=0)),0)
#                     # print(inputs)
#                     # print(torch.tensor(inlist))
#                     inputs = Variable(inlist.float(), requires_grad=True)
#
#                 self.optimizer.zero_grad()
#                 self.model.train()
#
#                 outputs = self.model(inputs)
#
#                 # print(outputs.grad, inputs.grad)
#                 # print(labels, outputs)
#
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#
#                 # print(inputs.grad)
#                 # print(loss.item())
#                 running_loss += loss.item()
#                 # for param in self.model.parameters():
#                 #     print(param)
#                 # print(outputs.grad, inputs.grad)
#                 # input()
#
#                 self.num_data += self.batch_size
#                 self.positive += len(torch.nonzero(labels, out=None))
#                 self.weight_pos = 1 - self.positive/self.num_data
#                 self.weight = torch.tensor([1-self.weight_pos, self.weight_pos]).cuda(torch.cuda.current_device())
#                 # self.criterion = nn.CrossEntropyLoss(weight=self.weight)
#                 # print(self.weight, self.weight_pos, self.positive)
#             print('train epoch loss:', running_loss/(i+1))
#         return running_loss/(i+1)
#
#     def predict(self, data):
#         # testset = TensorDataset(data)
#         if len(data.shape) == 1:
#             data = torch.unsqueeze(data, dim=0)
#         testloader = DataLoader(torch.tensor(data),
#                                 batch_size=self.batch_size,
#                                 shuffle=False)
#         loss = AverageRecorder()
#
#         # model = self.model.to(self.device)
#         self.model.eval()
#         outputs = []
#         preds = []
#         with torch.no_grad():
#             for i,data in enumerate(testloader):
#                 if self.cudaFlag:
#                     inputs = data
#                     if self.cudaFlag:
#                         inputs = inputs.cuda(device=torch.cuda.current_device())
#                         # labels = labels.cuda(device=torch.cuda.current_device())
#                 if self.batch_size == 1:
#                     inputs = Variable(torch.unsqueeze(inputs, dim=0).float(), requires_grad=True)
#                 else:
#                     inlist = None
#                     for i in inputs:
#                         if inlist is None:
#                             inlist = torch.unsqueeze(torch.unsqueeze(i, dim=0), dim=0)
#                         else:
#                             inlist = torch.cat((inlist,torch.unsqueeze(torch.unsqueeze(i, dim=0), dim=0)),0)
#                     inputs = Variable(inlist.float(), requires_grad=True)
#                 output = self.model(inputs)
#                 _, pred = torch.max(output, 1)
#                 # print(output, _, pred)
#                 for i in range(len(pred)):
#                     # outputs.append(torch.squeeze(output[i].cpu()))
#                     # preds.append(np.array(torch.squeeze(pred[i].cpu())))
#                     outputs.append(list(output[i].cpu()))
#                     preds.append(int(pred[i].cpu()))
#
#             # self.outputs = np.array([np.array(i) for i in outputs])
#             self.outputs = np.array(outputs)
#             self.preds = np.array(preds)
#             # self.preds = preds
#
#     def probablistic_matrix(self):
#         from scipy.special import softmax, expit
#         proba = self.outputs**3
#         proba_pred = np.zeros(self.outputs.shape)
#         for i in range(len(self.outputs)):
#             proba_pred[i] = softmax(proba[i])
#         return proba_pred

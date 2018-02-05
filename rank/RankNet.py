# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from tqdm import tqdm
from learning2rank.utils import plot_result
from learning2rank.utils import NNfuncs

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


class Model(chainer.Chain):
    """
    RankNet - Pairwise comparison of ranking.
    The original paper:
        https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf
    Japanese only:
        http://qiita.com/sz_dr/items/0e50120318527a928407
    """
    def __init__(self, n_in, n_units1, n_units2, n_out):
        super(Model, self).__init__(l1=L.Linear(n_in, n_units1),
                                    l2=L.Linear(n_units1, n_units2),
                                    l3=L.Linear(n_units2, n_out))

    def __call__(self, x_i, x_j, t_i, t_j):

        s_i = self.l3(F.relu(self.l2(F.relu(self.l1(x_i)))))
        s_j = self.l3(F.relu(self.l2(F.relu(self.l1(x_j)))))
        s_diff = s_i - s_j

        if t_i.data > t_j.data:
            S_ij = 1
        elif t_i.data < t_j.data:
            S_ij = -1
        else:
            S_ij = 0
        self.loss = (1 - S_ij) * s_diff / 2. + F.log(1 + F.exp(-s_diff))
        return self.loss

    def predict(self, x):

        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = F.relu(self.l3(h2))

        return h.data


class RankNet(NNfuncs.NN):
    """
    RankNet training function.
    Usage (Initialize):
        RankModel = RankNet()

    Usage (Traininng):
        Model.fit(X, y)

        Default options for fit function:
          - batchsize: 100
          - iterations: 5000
          - n_units1: 512
          - n_units2: 128
          - tv_ratio: 0.95
          - optimizer: 'Adam'
          - save_image_name: 'result.pdf'
          - save_model_name: 'RankNet.model'
    """
    def __init__(self, resume_model_name=None, verbose=True):
        self.resume_model_name = resume_model_name
        self.train_loss, self.test_loss = [], []
        self._verbose = verbose

        if resume_model_name is not None:
            print('Load resume model')
            self.load_model(resume_model_name)

    def ndcg(self, y_true, y_score, k=100):
        '''
        Evaluation function of NDCG@100
        '''
        y_true = y_true.ravel()
        y_score = y_score.ravel()
        y_true_sorted = sorted(y_true, reverse=True)
        ideal_dcg = 0
        for i in range(k):
            ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
        dcg = 0
        argsort_indices = np.argsort(y_score)[::-1]
        for i in range(k):
            dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
        ndcg = dcg / ideal_dcg

        return ndcg

    def train_model(self, x_train, y_train, x_test, y_test, iterations):
        '''
        Training function
        '''
        loss_step = 100

        for step in tqdm(range(iterations)):
            i, j = np.random.randint(len(x_train), size=2)
            x_i = chainer.Variable(x_train[i].reshape(1, -1))
            x_j = chainer.Variable(x_train[j].reshape(1, -1))
            y_i = chainer.Variable(y_train[i])
            y_j = chainer.Variable(y_train[j])
            self.optimizer.update(self.model, x_i, x_j, y_i, y_j)

            if (step + 1) % loss_step == 0:
                train_score = self.model.predict(chainer.Variable(x_train))
                test_score = self.model.predict(chainer.Variable(x_test))
                train_ndcg = self.ndcg(y_train, train_score)
                test_ndcg = self.ndcg(y_test, test_score)
                self.train_loss.append(train_ndcg)
                self.test_loss.append(test_ndcg)
                if self._verbose:
                    print('step: {0}'.format(step + 1))
                    print('NDCG@100 | train: {0}, test: {1}'.format(train_ndcg, test_ndcg))

    def fit(self, fit_X, fit_y, batchsize=100, iterations=5000, n_units1=512, n_units2=128,
            tv_ratio=0.95, optimizer_algorithm='Adam', save_image_name='Result.pdf',
            save_model_name='RankNet.model'):

        train_X, train_y, validate_X, validate_y = self.splitData(fit_X, fit_y, tv_ratio)
        print("The number of data, train:", len(train_X), "validate:", len(validate_X))

        if self.resume_model_name is None:
            self.initializeModel(Model, train_X, n_units1, n_units2, optimizer_algorithm)

        self.train_model(train_X, train_y, validate_X, validate_y, iterations)

        plot_result.acc(self.train_loss, self.test_loss, savename=save_image_name)
        self.save_models(save_model_name)

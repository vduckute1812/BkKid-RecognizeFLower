#!/usr/bin/env python

'''
Digit recognition adjustment.
Grid search is used to find the best parameters for SVM and KNearest classifiers.
SVM adjustment follows the guidelines given in
http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
Usage:
  digits_adjust.py [--model {svm|knearest}]
  --model {svm|knearest}   - select the classifier (SVM is the default)
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import math
import os
import cv2
import numpy as np
from setting import GRAY_PATH
from multiprocessing.pool import ThreadPool
from helper import huMonents, list_item, logarit


label = {
    "Dahlia": 0,
    "Ixora": 1,
    "Lily": 2,
    "Rose": 3,
    "Sun": 4,
    "Hoa But": 5,
    "Hoa Cam Tu Cau":6,
    "Hoa Canh Buom":7,
    "Hoa Cuc Trang":8,
    "Hoa Hong Mon":9,
    "Hoa Mao Ga":10,
    "Hoa Rum":11,
    "Hoa Sen":12,
    "Hoa Thien Dieu":13,
    "Hoa Van Tho":14
}

get_name = {
    0: "Dahlia",
    1: "Ixora",
    2: "Lily",
    3: "Rose",
    4: "Sun",
    5: "Hoa But",
    6: "Hoa Cam Tu Cau",
    7: "Hoa Canh Buom",
    8: "Hoa Cuc Trang",
    9: "Hoa Hong Mon",
    10:"Hoa Mao Ga",
    11:"Hoa Rum",
    12:"Hoa Sen",
    13:"Hoa Thien Dieu",
    14:"Hoa Van Tho"
}


def description(PATH_FLOWER):
    X = []
    y = []
    num_of_flower = {
        "Dahlia": 0,
        "Ixora": 0,
        "Lily": 0,
        "Rose": 0,
        "Sun": 0,
        "Hoa But": 0,
        "Hoa Cam Tu Cau": 0,
        "Hoa Canh Buom": 0,
        "Hoa Cuc Trang": 0,
        "Hoa Hong Mon": 0,
        "Hoa Mao Ga": 0,
        "Hoa Rum": 0,
        "Hoa Sen": 0,
        "Hoa Thien Dieu": 0,
        "Hoa Van Tho": 0
    }
    for index,item_path in enumerate(list_item(PATH_FLOWER)):
        name = os.path.split(os.path.dirname(item_path))[1]
        image = cv2.imread(item_path)
        #hu = huMonents(image)
        hu = logarit(image)
        num_of_flower[name] += 1
        X.append(hu)
        y.append(label[name])
    trainData=np.float32(X).reshape(-1,7)
    responses=np.int32(y).reshape(-1,1)
    return trainData,responses,num_of_flower


class SVM:
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def cross_validate(model_class, params, samples, labels, kfold = 150, pool = None):
    n = len(samples)
    folds = np.array_split(np.arange(n), kfold)
    def f(i):
        model = model_class(**params)
        test_idx = folds[i]
        train_idx = list(folds)
        train_idx.pop(i)
        train_idx = np.hstack(train_idx)
        train_samples, train_labels = samples[train_idx], labels[train_idx]
        test_samples, test_labels = samples[test_idx], labels[test_idx]
        model.train(train_samples, train_labels)
        resp = model.predict(test_samples)
        score = (resp != test_labels).mean() 
        print(".", end='')
        return score
    if pool is None:
        scores = list(map(f, range(kfold)))
    else:
        scores = pool.map(f, range(kfold))
    return np.mean(scores)

class SVM_OptimalParameter:
    def __init__(self, _samples, _labels):
        self._samples = _samples
        self._labels = _labels

    def get_dataset(self):
        return self._samples, self._labels

    def run_jobs(self, f, jobs):
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        ires = pool.imap_unordered(f, jobs)
        return ires

    def adjust_SVM(self):
        Cs = np.logspace(1, 15 , 20, base=2)
        gammas = np.logspace(-10, 0, 5, base=2)
        scores = np.zeros((len(Cs), len(gammas)))
        scores[:] = np.nan

        print('adjusting SVM (may take a long time) ...')
        def f(job):
            i, j = job
            samples, labels = self.get_dataset()
            #print(labels)
            params = dict(C = Cs[i], gamma=gammas[j])
            score = cross_validate(SVM, params, samples, labels)
            return i, j, score
        ires = self.run_jobs(f, np.ndindex(*scores.shape))
        for count, (i, j, score) in enumerate(ires):
            scores[i, j] = score
            print('%d / %d (best error: %.2f %%, last: %.2f %%)' %
                  (count+1, scores.size, np.nanmin(scores)*100, score*100))
        print(scores)

        print('writing score table to "svm_scores.npz"')
        np.savez('svm_scores.npz', scores=scores, Cs=Cs, gammas=gammas)

        i, j = np.unravel_index(scores.argmin(), scores.shape)
        best_params = dict(C = Cs[i], gamma=gammas[j])
        print('best params:', best_params)
        print('best error: %.2f %%' % (scores.min()*100))
        return best_params

    def adjust_KNearest(self):
        print('adjusting KNearest ...')
        def f(k):
            samples, labels = self.get_dataset()
            err = cross_validate(KNearest, dict(k=k), samples, labels)
            return k, err
        best_err, best_k = np.inf, -1
        for k, err in self.run_jobs(f, range(1, 9)):
            if err < best_err:
                best_err, best_k = err, k
            print('k = %d, error: %.2f %%' % (k, err*100))
        best_params = dict(k=best_k)
        print('best params:', best_params, 'err: %.2f' % (best_err*100))
        return best_params


if __name__ == '__main__':
    import getopt
    import sys
    import time

    print(__doc__)

    args, _ = getopt.getopt(sys.argv[1:], '', ['model='])
    args = dict(args)
    args.setdefault('--model', 'svm')
    args.setdefault('--env', '')
    if args['--model'] not in ['svm', 'knearest']:
        print('unknown model "%s"' % args['--model'])
        sys.exit(1)

    t = time.clock()
    #X,y,num_of_flower = description(GRAY_PATH)
    #app = SVM_OptimalParameter(X,y)
    #app.adjust_SVM()
    print('work time: %f s' % (time.clock() - t))

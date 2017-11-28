#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2017 CS6375
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = "Hanlin He (hxh160630), Mingze Xu (mxx160530), Su Yang (sxy161730) and Tao Wang (txw162630)"
__copyright__ = "Copyright 2017, The CS6375 Project"
__license__ = "MIT License"
__version__ = "1.0.0"
__maintainer__ = "Hanlin He"
__email__ = "hxh160630@utdallas.edu"
__status__ = "Development"

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from tqdm import tqdm

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = './data/sample_submission.csv'


def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g


if __name__ == '__main__':

    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %('
        'message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df = load_train_data()

    x_train = df.drop('target', axis=1)
    y_train = df['target'].values

    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(x_train.shape))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    all_params = {
        'loss': ['deviance', 'exponential'],
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4]
    }
    min_score = 100
    min_params = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        list_gini_score = []
        list_logloss_score = []
        all_preds = np.zeros(shape=y_train.shape[0])
        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            clf = GradientBoostingClassifier(**params)
            clf.fit(trn_x, trn_y)
            pred = clf.predict_proba(val_x)[:, 1]
            sc_logloss = log_loss(val_y, pred)
            sc_gini = - gini(val_y, pred)

            all_preds[valid_idx] = pred

            list_logloss_score.append(sc_logloss)
            list_gini_score.append(sc_gini)
            logger.debug('   logloss: {}, gini: {}'.format(sc_logloss, sc_gini))
            break

        sc_logloss = np.mean(list_logloss_score)
        sc_gini = np.mean(list_gini_score)
        if min_score > sc_gini:
            min_score = sc_gini
            min_params = params
        logger.info('logloss: {}, gini: {}'.format(sc_logloss, sc_gini))
        logger.info(
            'current min score: {}, params: {}'.format(min_score, min_params))

    logger.info('minimum params: {}'.format(min_params))
    logger.info('minimum gini: {}'.format(min_score))

    clf = GradientBoostingClassifier(**min_params)
    clf.fit(x_train, y_train)

    logger.info('train end')

    df = load_test_data()
    for col in use_cols:
        if col not in df.columns:
            logger.info('{} is not in test data'.format(col))
            df[col] = np.zeros(df.shape[0])
    x_test = df[use_cols].sort_values('id')

    logger.info('test data load end {}'.format(x_test.shape))
    pred_test = clf.predict_proba(x_test)[:, 1]

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test

    df_submit.to_csv(DIR + 'submit.csv', index=False)

    logger.info('end')

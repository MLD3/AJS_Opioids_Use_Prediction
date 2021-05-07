import numpy as np
import scipy.stats as st
import sklearn as sk
from sklearn.metrics import roc_auc_score, make_scorer, mean_squared_error, precision_recall_curve, average_precision_score
import pickle

path = 'X:/Students/Jaewon/FIDDLE_output_dt365/'

def get_training(outcome):
    path = '/scratch/filip_root/filip/jaewonh/dt365_new/train_test/'
    X_train = np.load(path + 'flat_train_quantity_recency.npy')
    if outcome == 'refill':
        y_train = np.load(path + 'refill_train.npy')
    elif outcome == 'pu':
        y_train = np.load(path + 'prolonged_use_train.npy')
    return X_train, y_train

def get_testing(outcome):
    path = '/scratch/filip_root/filip/jaewonh/dt365_new/train_test/'
    X_train = np.load(path + 'flat_test_quantity_recency.npy')
    if outcome=='refill':
        y_train = np.load(path + 'refill_test.npy')
    elif outcome=='pu':
        y_train = np.load(path + 'prolonged_use_test.npy')
    return X_train, y_train

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5)
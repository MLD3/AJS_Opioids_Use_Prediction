import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import sklearn as sk
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import scipy.stats as st
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, make_scorer, mean_squared_error, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve
import pickle
import copy
import joblib
import sys

class OpioidClassifier:
    #'Characterizes a dataset for PyTorch'
    def __init__(self, model, model_type):
        #'Initialization'
        self.model = model
        self.model_type = model_type

    def predict(self, X):
        if self.model_type=='xgb':
            return self.model.predict_proba(X, None, False)[:,1]
        elif self.model_type=='linear_svc':
            return self.model.decision_function(X)

def get_training(outcome):
    path = '/home/jaewonh/data/flat/train_test/'
    X_train = np.load(path + 'flat_train_quantity_recency.npy')
    if outcome == 'refill':
        y_train = np.load(path + 'refill_train.npy')
    elif outcome == 'pu':
        y_train = np.load(path + 'prolonged_use_train.npy')
    return X_train, y_train

def get_testing(outcome):
    path = '/home/jaewonh/data/flat/train_test/'
    X_train = np.load(path + 'flat_test_quantity_recency.npy')
    if outcome=='refill':
        y_train = np.load(path + 'refill_test.npy')
    elif outcome=='pu':
        y_train = np.load(path + 'prolonged_use_test.npy')
    return X_train, y_train

def eval_auc_score(classifier, X, y):
    y_score = classifier.predict(X)
    auc_score = roc_auc_score(y, y_score)
    return auc_score

def get_permuted_score_differences(X, y, true_score, indices, classifier):
    score_differences = []
    X_perm = np.copy(X)
    for _ in range(5):
        for col in indices:
            np.random.shuffle(X_perm[:,col])
        permutation_score = eval_auc_score(classifier, X_perm, y)
        score_differences.append(true_score - permutation_score)

    return score_differences

def get_all_ones_matrix(X, indices):
    X_all_ones = np.copy(X)
    for col in indices:
        X_all_ones[:,col] = np.ones(X_all_ones.shape[0])
    return X_all_ones

def save_dict_to_df(group_scores, list_of_related_features, features, name, model_name):
    df_dict = {"groups": [], "means": [], "stds": []}
    for group_index, mean_std in group_scores.items():
        group_indices = list_of_related_features[group_index]
        group = []
        for index in group_indices:
            group += [features[index]]
        df_dict["groups"].append(group)
        df_dict["means"].append(mean_std[0])
        df_dict["stds"].append(mean_std[1])
    
    group_scores_df = pd.DataFrame.from_dict(df_dict)
    group_scores_df.to_csv('/home/jaewonh/' + name + '_' + model_name + '.csv')
    
def group_importance_dict(classifier, X, y, features, model_name):
    with open('/home/jaewonh/no_bootstrapping_perm_imps/list_of_related_features.pkl', 'rb') as f:
        list_of_related_features = pickle.load(f)

    group_scores = {}
    group_effects = {}

    true_score = eval_auc_score(classifier, X, y)

    for group_index in range(len(list_of_related_features)):
        indices = list_of_related_features[group_index]
        ### Permute the related features and get score differences
        permuted_score_differences = get_permuted_score_differences(X, y, true_score, indices, classifier)
        
        mean, std = np.mean(permuted_score_differences), np.std(permuted_score_differences, ddof=1)
        group_scores[group_index] = [mean, std]     

    with open('/home/jaewonh/no_bootstrapping_perm_imps/no_bootstrap_related_group_imps_dict_' + model_name + '.pkl', 'wb') as f:
        pickle.dump(group_scores, f)

    save_dict_to_df(group_scores, list_of_related_features, features, 'no_bootstrap_correlated_group_imps_df', model_name)

### model_type should be 'linear_svc' or 'xgb'.
### outcome should be 'refill' or 'pu'
def get_permutation_importances(model_type, outcome):
    with open('/home/jaewonh/data/flat/train_test/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)

    name = model_type + '_' + outcome

    model = None

    if model_type == 'xgb':
        model = joblib.load('/home/jaewonh/data/flat/train_test/models/' + name)
    elif model_type == 'linear_svc':
        model = pickle.load(open('/home/jaewonh/data/flat/train_test/models/' + name + '.pickle', "rb"))
    
    classifier = OpioidClassifier(model, model_type)

    X, y = get_testing(outcome)
    
    group_importance_dict(classifier, X, y, features, name)
    
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--outcome', type=str, required=True)

args = parser.parse_args()

model_type = args.model_type
outcome = args.outcome

get_permutation_importances(model_type, outcome)
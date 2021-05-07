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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5)

### Consider scale_pos_weight!!!!
def get_CI_xgb(outcome):
    X_train, y_train = get_training(outcome)
    X_, y_ = get_testing(outcome)
    
    #X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, stratify=y_train, test_size=0.15, random_state=77)
    
    auc_scores = []
    ap_scores = []
        
    # Refill Params
    best_ntrees = 500
    best_depth = 9
    best_min_child_weight = 10
    best_gamma = 0
    best_pos_weight = 2
    best_subsample = 0.6
    best_colsample = 0.8
    best_lr = 0.03
    best_alpha = 0.2
    
    if outcome == 'pu':
        best_ntrees=1200
        best_depth = 5
        best_min_child_weight = 10
        best_gamma = 0.8
        best_pos_weight = 1
        best_subsample = 0.8
        best_colsample = 0.6
        best_lr = 0.01
        best_alpha = 0.3
    
    clf = xgb.XGBClassifier(n_estimators=best_ntrees,
                            max_depth=best_depth,
                            learning_rate=best_lr,
                            min_child_weight=best_min_child_weight,
                            gamma=best_gamma,
                            subsample=best_subsample,
                            colsample_bytree=best_colsample,
                            reg_alpha=best_alpha,
                            objective= 'binary:logistic',
                            n_jobs=1,
                            scale_pos_weight=best_pos_weight)
    
    #clf.fit(X_tr, y_tr,  eval_set=[(X_va,y_va)], eval_metric='auc', verbose=False, early_stopping_rounds=50)
    clf.fit(X_train, y_train)
    
    y_score = clf.predict_proba(X_)[:,1]
    
    dump_list = clf.get_booster().get_dump()
    num_trees = len(dump_list)
    
    print(num_trees)
    
    for i in range(1000):
        y_true_b, y_score_b = sk.utils.resample(y_, y_score, replace=True, random_state=i)
        
        ap_score = average_precision_score(y_true_b, y_score_b)
        auc_score = roc_auc_score(y_true_b, y_score_b)

        ap_scores.append(ap_score)
        auc_scores.append(auc_score)

    mean_auc, lower_auc, upper_auc = mean_confidence_interval(auc_scores, confidence=0.95)

    mean_ap, lower_ap, upper_ap = mean_confidence_interval(ap_scores, confidence=0.95)
    
    if outcome=='refill':
        # save
        joblib.dump(clf, '/scratch/filip_root/filip/jaewonh/dt365_new/train_test/models/xgb_refill')
    elif outcome == 'pu':
        joblib.dump(clf, '/scratch/filip_root/filip/jaewonh/dt365_new/train_test/models/xgb_pu')
        
    print("AUC {}, ({}, {})".format(mean_auc, lower_auc, upper_auc))
    print("AP {}, ({}, {})".format(mean_ap, lower_ap, upper_ap))

    return clf

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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

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

def eval_auc_score(classifier, X, y):
    y_score = classifier.predict(X)
    auc_score = roc_auc_score(y, y_score)
    return auc_score

def get_permuted_matrix(X, indices):
    X_perm = np.copy(X)
    for col in indices:
        np.random.shuffle(X_perm[:,col])
    return X_perm

def get_all_ones_matrix(X, indices):
    X_all_ones = np.copy(X)
    for col in indices:
        X_all_ones[:,col] = np.ones(X_all_ones.shape[0])
    return X_all_ones

def save_dict_to_df(group_scores, correlated_groups_list, features, name, model_name):
    df_dict = {"groups": [], "scores": [], "lbs": [], "ubs": []}
    for group_index, CI in group_scores.items():
        group_indices = correlated_groups_list[group_index]
        group = []
        for index in group_indices:
            group += [features[index]]
        df_dict["groups"].append(group)
        df_dict["scores"].append(CI[0])
        df_dict["lbs"].append(CI[1])
        df_dict["ubs"].append(CI[2])
    
    group_scores_df = pd.DataFrame.from_dict(df_dict)
    group_scores_df.to_csv('/home/jaewonh/' + name + '_' + model_name + '.csv')
    
def corr_importance_dict(classifier, X, y, features, model_name):
    with open('/home/jaewonh/list_of_related_features.pkl', 'rb') as f:
        correlated_groups_list = pickle.load(f)

    group_scores = {}
    group_effect = {}

    y_actual = classifier.predict(X)

    for group_index in range(len(correlated_groups_list)):
        ### Get permuted matrix
        group_indices = correlated_groups_list[group_index]
        X_perm = get_permuted_matrix(X, group_indices)
        ### Classify using model
        y_perm = classifier.predict(X_perm)
        ### 1000 resample
        auc_score_diffs = []
        for i in range(1000):
            samp_true, samp_perm, samp_actual = sk.utils.resample(y, y_perm, y_actual, replace=True, random_state=i)

            auc_actual = roc_auc_score(samp_true, samp_actual)
            auc_perm = roc_auc_score(samp_true, samp_perm)
            
            auc_score_diffs.append(auc_actual - auc_perm)
        
        mean, lb, ub = mean_confidence_interval(auc_score_diffs)
        group_scores[group_index] = [mean, lb, ub]
        
        ### Compute Group Effect
        X_ones = get_all_ones_matrix(X, group_indices)
        y_ones = classifier.predict(X_ones)
        diff_in_pos = []
        for i in range(1000):
            samp_ones, samp_actual = sk.utils.resample(y_ones, y_actual, replace=True, random_state=i)
        
            ones_pos_count = np.sum(samp_ones)
            actual_pos_count = np.sum(samp_actual)
            diff_in_pos.append(ones_pos_count - actual_pos_count)

        mean_pos, lb_pos, ub_pos = mean_confidence_interval(diff_in_pos)
        group_effect[group_index] = [mean_pos, lb_pos, ub_pos]
            

    with open('/home/jaewonh/correlated_group_imps_dict_related_feats' + model_name + '.pkl', 'wb') as f:
        pickle.dump(group_scores, f)

    with open('/home/jaewonh/correlated_group_effects_dict_related_feats' + model_name + '.pkl', 'wb') as f:
        pickle.dump(group_effect, f)

    save_dict_to_df(group_scores, correlated_groups_list, features, 'correlated_group_imps_df_related_feats', model_name)
    save_dict_to_df(group_effect, correlated_groups_list, features, 'correlated_group_effects_df_related_feats', model_name)

    

### model_type should be 'linear_svc' or 'xgb'.
### outcome should be 'refill' or 'pu'
def get_permutation_importances(model_type, outcome):
    with open('/scratch/filip_root/filip/jaewonh/dt365_new/train_test/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)

    name = model_type + '_' + outcome

    model = None

    if model_type == 'xgb':
        model = joblib.load('/scratch/filip_root/filip/jaewonh/dt365_new/train_test/models/' + name)
    elif model_type == 'linear_svc':
        model = pickle.load(open('/scratch/filip_root/filip/jaewonh/dt365_new/train_test/models/' + name + '.pickle', "rb"))
    
    classifier = OpioidClassifier(model, model_type)

    X, y = get_testing(outcome)
    
    corr_importance_dict(classifier, X, y, features, name)
    
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--outcome', type=str, required=True)

args = parser.parse_args()

model_type = args.model_type
outcome = args.outcome

get_permutation_importances(model_type, outcome)
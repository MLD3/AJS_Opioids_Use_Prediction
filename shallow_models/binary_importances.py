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

### Expects all_features in order of columns in the training/testing matrix
### Expects feature_names in list of list format e.g. [ [feat_A], [feat_B, feat_C], ... ]
### Converts to [ [feat_A_col_index], [feat_B_col_index, feat_C_col_index], ... ]
def convert_feature_names_to_indices(all_features, feature_names):
    feature_indices = []

    name_to_index = {}
    for i in range(len(all_features)):
        name_to_index[all_features[i]] = i

    for name_group in feature_names:
        indices = []
        for name in name_group:
            indices.append(name_to_index[name])
        feature_indices.append(indices)

    return feature_indices

### Expects two indices: one yes, one no. Flips all the yes entries to 1, all the no entries to 0.
def get_binary_ones_matrix(X, yes_index, no_index):
    X_bin = np.copy(X)
    X_bin[:,yes_index] = np.ones(X_bin.shape[0])
    X_bin[:, no_index] = np.zeros(X_bin.shape[0])
    return X_bin

### Expects trained classifier (e.g. xgb classifier, linear svc)
###         X, y data to evaluate on
###         all_features, the list of all features [feature_A, feature_B, feature_C, ...] in order
###         imp_features, the list of lists of correlated binary important features e.g. [ [feat_A_no, feat_A_yes], [feat_B_no, feat_B_yes], ... ]
###         model_outcome, to label the effects with, e.g. "xgb_pu"
def get_binary_effects_dict(classifier, X, y, all_features, imp_features, model_outcome):
    
    correlated_groups_list = convert_feature_names_to_indices(all_features, imp_features)

    group_effect = {}

    y_actual = classifier.predict(X)

    for group_index in range(len(correlated_groups_list)):

        group_indices = correlated_groups_list[group_index]

        no_index = group_indices[0]
        yes_index = group_indices[1]
        ### Compute Group Effect
        X_ones = get_binary_ones_matrix(X, yes_index, no_index)
        y_ones = classifier.predict(X_ones)
        diff_in_pos = []
        for i in range(100):
            samp_ones, samp_actual = sk.utils.resample(y_ones, y_actual, replace=True, random_state=i)
        
            ones_pos_count = np.sum(samp_ones)
            actual_pos_count = np.sum(samp_actual)
            diff_in_pos.append(ones_pos_count - actual_pos_count)

        mean_pos, lb_pos, ub_pos = mean_confidence_interval(diff_in_pos)
        group_effect[group_index] = [mean_pos, lb_pos, ub_pos]

    save_dict_to_df(group_effect, correlated_groups_list, all_features, 'correlated_group_effects_only_imp_df_100', model_outcome)

### model_type should be 'linear_svc' or 'xgb'.
### outcome should be 'refill' or 'pu'
def get_binary_effects(model_type, outcome, binary_features):
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

    get_binary_effects_dict(classifier, X, y, features, binary_features, name)

binary_top_xgb_pu_features = [['Spondylosis; intervertebral disc disorders; other back problems_value_n', 'Spondylosis; intervertebral disc disorders; other back problems_value_y'],
['Other non-traumatic joint disorders_value_n', 'Other non-traumatic joint disorders_value_y'],
['Arthritis and joint pain_value_n', 'Arthritis and joint pain_value_y'],
['Uses tobacco_value_n', 'Uses tobacco_value_y'],
['Charlson Comorbidity Index_value_(-0.001, 1.0]', 'Charlson Comorbidity Index_value_(1.0, 19.0]'],
['Headache; including migraine_value_n', 'Headache; including migraine_value_y'],
['Preop benzodiazepine use_value_n', 'Preop benzodiazepine use_value_y'],
['Headache/head pain/suboccipital headache_value_n', 'Headache/head pain/suboccipital headache_value_y'],
['Osteoarthritis_value_n', 'Osteoarthritis_value_y'],
['surg_gp_value_Minor Surgery', 'surg_gp_value_Major Surgery'],]

binary_top_xgb_refill_features = [['surg_gp_value_Minor Surgery', 'surg_gp_value_Major Surgery'],
['Opioid prescription within 30 days prior surgery/admission_value_n', 'Opioid prescription within 30 days prior surgery/admission_value_y'],
['Abdominal pain_value_n', 'Abdominal pain_value_y'],
['Uses tobacco_value_n', 'Uses tobacco_value_y'],
['Functional bowel (includes IBS)_value_n', 'Functional bowel (includes IBS)_value_y'],
['Is an inpatient_value_n', 'Is an inpatient_value_y'],
['gender_value_F', 'gender_value_M'],
['Preop benzodiazepine use_value_n', 'Preop benzodiazepine use_value_y'],
['Anxiety disorders_value_n', 'Anxiety disorders_value_y'],
['Mood disorders_value_n', 'Mood disorders_value_y'],]

binary_top_linear_svc_pu_features = [['Spondylosis; intervertebral disc disorders; other back problems_value_n', 'Spondylosis; intervertebral disc disorders; other back problems_value_y'],
['Other non-traumatic joint disorders_value_n', 'Other non-traumatic joint disorders_value_y'],
['Charlson Comorbidity Index_value_(-0.001, 1.0]', 'Charlson Comorbidity Index_value_(1.0, 19.0]'],
['Arthritis and joint pain_value_n', 'Arthritis and joint pain_value_y'],
['Preop benzodiazepine use_value_n', 'Preop benzodiazepine use_value_y'],
['Is an inpatient_value_n', 'Is an inpatient_value_y'],
['Uses tobacco_value_n', 'Uses tobacco_value_y'],
['Headache/head pain/suboccipital headache_value_n', 'Headache/head pain/suboccipital headache_value_y'],
['Back (thoracic, lumbar, sacral spine)_value_n', 'Back (thoracic, lumbar, sacral spine)_value_y'],
['Osteoarthritis_value_n', 'Osteoarthritis_value_y'],
['Headache; including migraine_value_n', 'Headache; including migraine_value_y'],
['Abdominal pain_value_n', 'Abdominal pain_value_y'],
['gender_value_F', 'gender_value_M'],
['Fibromyalgia_value_n', 'Fibromyalgia_value_y'],
['Noncardiac or nonmusculoskeletal chest pain_value_n', 'Noncardiac or nonmusculoskeletal chest pain_value_y']
]

binary_top_linear_svc_refill_features = [['surg_gp_value_Minor Surgery', 'surg_gp_value_Major Surgery'],
['Opioid prescription within 30 days prior surgery/admission_value_n', 'Opioid prescription within 30 days prior surgery/admission_value_y'],
['Uses tobacco_value_n', 'Uses tobacco_value_y'],
['Functional bowel (includes IBS)_value_n', 'Functional bowel (includes IBS)_value_y'],
['Preop benzodiazepine use_value_n', 'Preop benzodiazepine use_value_y'],
['gender_value_F', 'gender_value_M'],
['Screening and history of mental health and substance abuse codes_value_n', 'Screening and history of mental health and substance abuse codes_value_y'],
['Other non-traumatic joint disorders_value_n', 'Other non-traumatic joint disorders_value_y'],
['Abdominal pain_value_n', 'Abdominal pain_value_y'],
['Insomnia_value_n', 'Insomnia_value_y'],]

get_binary_effects('linear_svc', 'pu', binary_top_linear_svc_pu_features)

get_binary_effects('linear_svc', 'refill', binary_top_linear_svc_refill_features)

get_binary_effects('xgb', 'pu', binary_top_xgb_pu_features)

get_binary_effects('xgb', 'refill', binary_top_xgb_refill_features)
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import sklearn as sk
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from scipy import sparse
import scipy.stats as st
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, make_scorer, mean_squared_error, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve
import pickle
import copy
import utils
#import joblib

def xgb_experiment(model_type, X, y):    
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size = 0.20, random_state = 7)
    
    fit_params={'early_stopping_rounds': 100, 
            'eval_metric': 'auc',
            'verbose': False,
            'eval_set': [[X_va, y_va]]}
    
    # grid search
    min_child_weights = [i*10 for i in range(0, 5)]
    max_depths = [3, 5, 7, 9, 11, 13]
    
    gammas = [i*0.4 for i in range(0, 4)]
    
    subsamples = [i/10.0 for i in range(4,10)]
    colsamples = [i/10.0 for i in range(4,10)]
        
    best_pos_weight = 1
    best_depth_score = -1
    best_depth = -1
    best_min_child_weight = -1
    best_gamma = -1
    
    for depth in max_depths:
        for min_child_weight in min_child_weights:
            for gamma in gammas:
                clf = xgb.XGBClassifier(n_estimators=1200,
                            max_depth=depth,
                            learning_rate=0.01,
                            min_child_weight=min_child_weight,
                            gamma=gamma,
                            subsample=0.7,
                            colsample_bytree=0.7,
                            objective= 'binary:logistic',
                            n_jobs=1,
                            scale_pos_weight=best_pos_weight)
                scores = cross_val_score(clf, X_tr, y_tr, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=77777), scoring='roc_auc', fit_params=fit_params)
                avg_score = mean(scores)
                print("depth = {}, min_child_weight = {}, gamma = {}, score = {}".format(depth, min_child_weight, gamma, avg_score))

                if (avg_score > best_depth_score):
                    best_depth_score = avg_score
                        
                    best_depth = depth
                    best_min_child_weight = min_child_weight
                    best_gamma = gamma
            
    best_subcol_score = -10000
    best_colsample = -1
    best_subsample = -1
    for subsample in subsamples:
        for colsample in colsamples:
            clf = xgb.XGBClassifier(n_estimators=1200,
                            max_depth=best_depth,
                            learning_rate=0.01,
                            min_child_weight=best_min_child_weight,
                            gamma=best_gamma,
                            subsample=subsample,
                            colsample_bytree=colsample,
                            objective= 'binary:logistic',
                            n_jobs=1,
                            scale_pos_weight=best_pos_weight)
            scores = cross_val_score(clf, X_tr, y_tr, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=77777), scoring='roc_auc', fit_params=fit_params)
            avg_score = mean(scores)
            print("subsample = {}, colsample = {}, score = {}".format(subsample, colsample, avg_score))

            if (avg_score > best_subcol_score):
                best_subsample = subsample
                best_colsample = colsample
                best_subcol_score = avg_score

    lrs = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5]
    alphas_init = [0.05, 0.1,  0.2, 0.3, 0.4, 0.5]

    best_score = -10000
    best_lr = -1
    best_alpha = -1
    for alpha in alphas_init:
        for lr in lrs:
            clf = xgb.XGBClassifier(n_estimators=1200,
                            max_depth=best_depth,
                            learning_rate=lr,
                            min_child_weight=best_min_child_weight,
                            gamma=best_gamma,
                            subsample=best_subsample,
                            colsample_bytree=best_colsample,
                            reg_alpha=alpha,
                            objective= 'binary:logistic',
                            n_jobs=1,
                            scale_pos_weight=best_pos_weight)
            scores = cross_val_score(clf, X_tr, y_tr, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=77777), scoring='roc_auc', fit_params=fit_params)
            avg_score = mean(scores)
            print("alpha: %s, score: %s", (alpha, avg_score))
            if (avg_score > best_score):
                best_alpha = alpha
                best_lr = lr
                best_score = avg_score

    print("Best depth = " + str(best_depth))
    print("Best min_child_weight = " + str(best_min_child_weight))
    print("Best gamma = " + str(best_gamma))
    print("Best pos_weight  {}".format(best_pos_weight))
    print("Best subsample = " + str(best_subsample))
    print("Best colsample = " + str(best_colsample))
    print("best lr = {}".format(best_lr))
    print("best alpha = {}".format(best_alpha))
    print("best score = {}".format(best_score))

    best_clf = xgb.XGBClassifier(n_estimators=1200,
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

    return best_clf

def linear_experiment(model_type, X, y):    
    Cs = [i*0.00005 for i in range(1,21)]
    
    dicts = []
    for i in range(1, 6):
        balance_dict = {0 : 1-0.1*i, 1 : 0.1*i}
        dicts.append(balance_dict)
    
    best_auc_C = -1
    best_dict = None
    top_auc = -1
    
    for c in Cs:
        for balance_dict in dicts:
            skf = StratifiedKFold(n_splits=10)
            aucs = []
            for train_index, test_index in skf.split(X, y):
                clf = LinearSVC(C=c, dual=False, class_weight=balance_dict)
                X_tr = X[train_index]
                y_tr = y[train_index]

                X_va = X[test_index]
                y_va = y[test_index]

                clf.fit(X_tr,y_tr)

                pred = clf.decision_function(X_va)
                auc_score = roc_auc_score(y_va, pred)

                aucs.append(auc_score)
        
            avg_auc_score = np.mean(aucs)
        
            if avg_auc_score > top_auc:
                top_auc = avg_auc_score
                best_dict = balance_dict
                best_auc_C = c
            
            print('Curr C is: ' + str(c))
            print(balance_dict)
            print("C and balance dict avg auc score is: " + str(avg_auc_score))
            
    print('Top auc score is: ' + str(top_auc))
    print('Best auc C is: ' + str(best_auc_C))
    print(best_dict)

    return LinearSVC(C=best_auc_C, dual=False, class_weight=best_dict)

def experiment(model_type, X, y):
    if model_type == 'linear_svc':
        return linear_experiment(model_type, X, y)
    elif model_type == 'xgb':
        return xgb_experiment(model_type, X, y)

class ShallowClassifier:
    #'Characterizes a dataset for PyTorch'
    def __init__(self, model=None, model_type, outcome):
        #'Initialization'
        self.model = model
        self.model_type = model_type
        self.outcome = outcome

    def load_existing(self, path):
        name = self.model_type + '_' + self.outcome
        
        if self.model_type == 'xgb':
            self.model = joblib.load(path + name)
        
        elif self.model_type == 'linear_svc':
            self.model = pickle.load(open(path + name + '.pickle', "rb"))

    def save_model(self, path):
        name = self.model_type + '_' + self.outcome
        
        if self.model_type == 'xgb':
            joblib.dump(self.model, path + name)
        
        elif self.model_type == 'linar_svc':
            pickle.dump(clf, open(path + name, "wb"))

    def predict(self):
        X, y = utils.get_testing(self.outcome)
        if self.model_type=='xgb':
            return self.model.predict_proba(X, None, False)[:,1]
        elif self.model_type=='linear_svc':
            return self.model.decision_function(X)

    def save_predictions(self, path):
        y_score = self.predict()
        name = self.model_type + '_' + self.outcome
        np.save(path + name + '_y_score', y_score)

    def experiment_and_set_hyperparameters(self):
        X, y = utils.get_training(self.outcome)
        self.model = experiment(self.model_type, X, y)

    def train(self):
        X, y = utils.get_training(self.outcome)

        if self.model == None:
            print("MODEL HAS NOT BEEN SET! RUN experiment_and_set_hyperparameters() before!")

        if self.model_type == 'linear_svc':
            self.model.fit(X, y)
        elif self.model_type == 'xgb':
            self.model.fit(X_train, y_train)


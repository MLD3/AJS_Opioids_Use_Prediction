import pandas as pd
import numpy as np
import sparse
import json

# get path from argument
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataframe_path', type=str, required=True)
parser.add_argument('--quantity_path', type=str, required=True)
parser.add_argument('--recency_path', type=str, required=True)
parser.add_argument('--target_path', type=str, required=True)
args = parser.parse_args()
dataframe_path = args.dataframe_path
quantity_path = args.dataframe_path
recency_path = args.patient_fill_info_path
target_path = args.target_path

### Get arrays
## Time dependent
X = sparse.load_npz(quantity_path + 'X.npz').todense()
## Time invariant
S = sparse.load_npz(quantity_path + 's_all.npz').todense()
## Recency data
R = sparse.load_npz(recency_path + 'X.npz').todense()

def get_shallow_split(X, S):
    ### Squish the 3D X into a 2D array, and insert S as well.
    full_data = np.zeros((X.shape[0], X.shape[2] + S.shape[1] + R.shape[2]))
    for row in range(full_data.shape[0]):
        ### Fill in the time dependent values first
        full_data[row, 0:X.shape[2]] = X[row, 0, :]
        ### Fill in the time invariant values next
        full_data[row, X.shape[2]:(S.shape[1] + X.shape[2])] = S[row,:]
        ### Fill in recency values last
        full_data[row, (X.shape[2] + S.shape[1]):full_data.shape[1]] = R[row, 0, :]
    
    ### Get train and test set patids via the following dfs:
    naive_train = pd.read_pickle(dataframe_path + 'naive_train.pkl')
    naive_test = pd.read_pickle(dataframe_path + 'naive_test.pkl')
    patids = naive_train.patid.tolist() + naive_test.patid.tolist()
    patids.sort()

    ### Map each index into a train or test set.
    train_indices = []
    test_indices = []

    for i in range(len(patids)):
        if patids[i] in naive_train.patid.values:
            train_indices.append(i)
        elif patids[i] in naive_test.patid.values:
            test_indices.append(i)
    
    ### This will not do anything if quintiled, but if values are continuous, it will make
    ### the values 0 to 1
    for feat_col in range(full_data.shape[1]):
        max_val = np.max(full_data[:,feat_col])
        full_data[:, feat_col] = np.divide(full_data[:, feat_col], max_val)

    train_set = full_data[train_indices, :]
    test_set = full_data[test_indices, :]

    np.save(path + 'flat_train', train_set)
    np.save(path + 'flat_test', test_set)

def get_deep_split():
    ### Append S to the X array (shortfuse)
    full_data = np.zeros((X.shape[0], X.shape[1], X.shape[2] + S.shape[1]))
    for row in range(full_data.shape[0]):
        for dt in X.shape[1]:
            ### Fill in the time dependent values first
            full_data[row, dt, 0:X.shape[2]] = X[row, dt, :]
            ### Fill in the time invariant values next
            full_data[row, dt, X.shape[2]:(X.shape[2] + S.shape[1]) ] = S[row,:]
    
    ### This will not do anything if quintiled, but if values are continuous, it will make
    ### the values 0 to 1
    for feat_col in range(full_data.shape[2]):
        max_val = np.max(full_data[:, :, feat_col])
        full_data[:, :, feat_col] = np.divide(full_data[:, :, feat_col], max_val)
    
    ### Get train and test set patids via the following dfs:
    naive_train = pd.read_pickle(dataframe_path + 'naive_train.pkl')
    naive_test = pd.read_pickle(dataframe_path + 'naive_test.pkl')
    patids = naive_train.patid.tolist() + naive_test.patid.tolist()
    patids.sort()

    ### Map each index into a train or test set.
    train_indices = []
    test_indices = []

    for i in range(len(patids)):
        if patids[i] in naive_train.patid.values:
            train_indices.append(i)
        elif patids[i] in naive_test.patid.values:
            test_indices.append(i)

    train_set = full_data[train_indices, :, :]
    test_set = full_data[test_indices, :, :]

    np.save(target_path + 'time_series_train', train_set)
    np.save(target_path + 'time_series_test', test_set)
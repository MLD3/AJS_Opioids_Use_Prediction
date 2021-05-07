import numpy as np
import pickle
import scipy.stats as st


data_path = '/home/jaewonh/data/flat/train_test/'
X_train = np.load(data_path + 'flat_train_quantity_recency.npy')
X_test = np.load(data_path + 'flat_test_quantity_recency.npy')
X_arr = np.concatenate((X_train, X_test), axis=0)

groups = [[0]]

for i in range(1, X_arr.shape[1]):
    found = False
    for j in range(len(groups)):
        rep = groups[j][0]
        if abs(st.pearsonr( X_arr[:,rep], X_arr[:, i])[0]) > 0.9:
            found = True
            groups[j].append(i)
            break
    if not found:
        groups.append([i])
    print("{} of {} features done.".format(i, X_arr.shape[1]))

print('number of groups is: ' + str(len(groups)))

for group in groups:
    print("\n")
    group_mems = ""
    for elt in group:
        group_mems += str(elt) + ", "
    print(group_mems)

with open('/home/jaewonh/correlated_groups_list_new_feat.pkl', 'wb') as f:
    pickle.dump(groups, f)
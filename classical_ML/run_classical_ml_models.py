# Load best parameters from text file
import pickle
from joblib import dump, load
import pandas as pd
import numpy as np
from tqdm import tqdm
from train_test_tune_ica import calc_f2_score
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from classical_ml_models import *

def run_classical_ml_models(path, train_folds, test_folds, data, labels, patient_id, models, feature_select):

    ## Reshape data
    # num_segments = data.shape[0]
    # num_channels = data.shape[1]
    # num_features = data.shape[2]

    # data_reshape = np.reshape(data, (num_segments, num_channels*num_features))

    # models = ['svc', 'rf', 'xg', 'gmm']
    # feature_select = ['umap', 'ica']
    # models = ['rf']
    # feature_select = ['umap']

    for model in models:
        for method in feature_select:
            print(model)
            print(method)
            print(' ')

            # Best params according to mean F2 score
            with open(path + model + '_best_params_' + method + '.pkl', 'rb') as f:
                best_params = pickle.load(f)

            # Estimator trained on all the data using parameters from the best F2 score
            # best_estimator = load(path + model + '_best_estimator_' + method + '.joblib')

            f2_scores = []
            acc_scores = []
            true_neg = []
            false_pos = []
            false_neg = []
            true_pos = []

            count = 1
            print('Running ', model, 'in CV ', method)
            for (train_idx, test_idx) in zip(train_folds, test_folds):
                X_train, X_test = data[train_idx], data[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
                group_train = patient_id[train_idx]

                print('Run: ', count)

                if model == 'svc':
                    y_pred_test, y_pred_proba = svm_model(X_train, y_train, X_test, best_params)
                elif model == 'rf':
                    y_pred_test, y_pred_proba = random_forest_model(X_train, y_train, X_test, best_params)
                elif model == 'gmm':
                    y_pred_test, y_pred_proba = gmm_model(X_train, y_train, X_test, best_params)
                elif model == 'xg': 
                    y_pred_test, y_pred_proba = xg_boost_model(X_train, y_train, X_test, best_params)
                else:
                    return

                # y_pred_test = best_estimator.predict(X_test)
                accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
                precision_test = precision_score(y_test, y_pred_test)
                recall_test = recall_score(y_test, y_pred_test)
                f2_test = calc_f2_score(precision_test, recall_test, 2)

                cm = confusion_matrix(y_test, y_pred_test)

                f2_scores.append(f2_test)
                acc_scores.append(accuracy_test)
                true_neg.append(cm[0][0])
                false_pos.append(cm[0][1])
                false_neg.append(cm[1][0])
                true_pos.append(cm[1][1])

                count = count + 1

            print('Saving scores')
            with open(path + model + '_fold_cm_scores_' + method + '.txt', 'w') as file:
                # Model and method
                file.write('Model: ' + model + ' Method: ' + method + '\n')
                file.write('Fold scores of best parameter set based on F2 scoring\n')

                file.write('Best F2 Parameter\n')
                for key, value in best_params.items():
                    file.write('%s: %s\n' % (key, value))
                # Write F2 scores
                file.write('\nF2 Scores\n')
                for item in f2_scores:
                    file.write('%s\n' % (item))

                # Write accuracy scores
                file.write('\nBalanced Accuracy Scores\n')
                for item in acc_scores:
                    file.write('%s\n' % (item))

                # Write true neg scores
                file.write('\nTrue Negative\n')
                for item in true_neg:
                    file.write('%s\n' % (item))

                # Write false pos scores
                file.write('\nFalse Positive\n')
                for item in false_pos:
                    file.write('%s\n' % (item))

                # Write true neg scores
                file.write('\nFalse Negative\n')
                for item in false_neg:
                    file.write('%s\n' % (item))

                # Write true neg scores
                file.write('\nTrue Positive\n')
                for item in true_pos:
                    file.write('%s\n' % (item))
            


save_path = '/raid/smtam/results/'
data_file_path = '/radraid/kpejcinovic/data/'

print("Loading in info files")
with open(save_path + 'patient_ID.pkl', 'rb') as f:
    patient_id = pickle.load(f)

with open(save_path + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# with open(save_path + 'data_reshape.pkl', 'rb') as f:
#     data_reshape = pickle.load(f)

print("Loading in features")
with open(data_file_path + 'features_3d_array.pkl', 'rb') as f:
    features_3d_array = pickle.load(f)

print("Train features array", features_3d_array.shape)

# splits = 5
# strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True, random_state=10)
# strat_kfold = strat_kfold_object.split(data_reshape, patient_id)

# train_total = []
# test_total = []

# print('Saving indices')

# for i, (train_idx, test_idx) in enumerate(strat_kfold):
#     train_total.append(train_idx)
#     test_total.append(test_idx)

# with open(save_path + 'stratkfold_train_idx.pkl', 'wb') as f:
#     pickle.dump(train_total, f)

# with open(save_path + 'stratkfold_test_idx.pkl', 'wb') as f:
#     pickle.dump(test_total, f)

print('Loading indices')

with open(save_path + 'stratkfold_train_idx.pkl', 'rb') as f:
    train_total = pickle.load(f)

with open(save_path + 'stratkfold_test_idx.pkl', 'rb') as f:
    test_total = pickle.load(f)

models = ['svc']
feature_select = ['ica', 'umap']

save_path = '/raid/smtam/results/tuning_results/'

run_classical_ml_models(save_path, train_total, test_total, features_3d_array, labels, patient_id, models, feature_select)


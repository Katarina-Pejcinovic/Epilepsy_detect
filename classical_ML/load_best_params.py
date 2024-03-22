
# Load best parameters from text file
import pickle
from joblib import dump, load
import pandas as pd
import numpy as np
from train_test_tune_ica import calc_f2_score

# def load_best_params():

#     with open('results/best_params_dict.pkl', 'rb') as f:
#         best_params = pickle.load(f)

#     print(best_params)

#     return best_params


# t = load_best_params()
# params_best = [{'svc__C': 1, 'svc__kernel': 'linear', 'umap__n_components': 3, 'umap__n_neighbors': 5}, {'randomforestclassifier__max_features': 25, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__n_estimators': 10, 'umap__n_components': 3, 'umap__n_neighbors': 5}, {'umap__n_components': 3, 'umap__n_neighbors': 10, 'xgbclassifier__learning_rate': 0.01, 'xgbclassifier__max_depth': 2, 'xgbclassifier__n_estimators': 50}, {'gaussianmixture__init_params': 'k-means++', 'umap__n_components': 3, 'umap__n_neighbors': 5}]

path = '/raid/smtam/results/tuning_results/'

# models = ['svc', 'rf', 'xg', 'gmm']
# feature_select = ['umap', 'ica']
models = ['xg']
feature_select = ['ica']

for model in models:
    for method in feature_select:
        print(model)
        print(method)
        print(' ')

        # Best params according to mean F2 score
        # with open(path + model + '_best_params_' + method + '.pkl', 'rb') as f:
        #     best_params_f2 = pickle.load(f)

        # # Mean F2, precision, recall, accuracy for the best parameter set (based on F2)
        # with open(path + model + '_best_scores_' + method + '.pkl', 'rb') as f:
        #     best_scores_f2 = pickle.load(f)
        
        print('Loading in results')
        # Get CV results - use to find F2 scores per fold, accuracies per fold of parameter set based on accuracy
        with open(path + model + '_cv_results_' + method + '.pkl', 'rb') as f:
            cv_results = pickle.load(f)
        
        cv_results_df = pd.DataFrame(cv_results)

        # Estimator trained on all the data using parameters from the best F2 score
        # f2_estimator = load(path + model + '_best_estimator_' + method + '.joblib')

        print('Finding scores')
        # Find the F2 scores from each split of the best parameter set (based on F2 for now)
        mean_accuracy = cv_results['mean_test_Accuracy']
        mean_precision = cv_results['mean_test_Precision']
        mean_recall = cv_results['mean_test_Recall']
        mean_f2 = calc_f2_score(mean_precision, mean_recall, 2)
        best_f2_index = np.nanargmax(mean_f2)
        best_accuracy_index = np.nanargmax(mean_accuracy)

        best_params_f2 = cv_results['params'][best_f2_index]
        best_score_f2 = [mean_f2[best_f2_index], mean_precision[best_f2_index], mean_recall[best_f2_index], 
                         mean_accuracy[best_f2_index]]
        
        best_params_acc = cv_results['params'][best_accuracy_index]
        best_score_acc = [mean_f2[best_accuracy_index], mean_precision[best_accuracy_index], mean_recall[best_accuracy_index], 
                         mean_accuracy[best_accuracy_index]]
        
        # Based on best F2
        f2_split0 = calc_f2_score(cv_results['split0_test_Precision'], cv_results['split0_test_Recall'], 2)[best_f2_index]
        f2_split1 = calc_f2_score(cv_results['split1_test_Precision'], cv_results['split1_test_Recall'], 2)[best_f2_index]
        f2_split2 = calc_f2_score(cv_results['split2_test_Precision'], cv_results['split2_test_Recall'], 2)[best_f2_index]
        f2_split3 = calc_f2_score(cv_results['split3_test_Precision'], cv_results['split3_test_Recall'], 2)[best_f2_index]
        f2_split4 = calc_f2_score(cv_results['split4_test_Precision'], cv_results['split4_test_Recall'], 2)[best_f2_index]

        f2_split = [f2_split0, f2_split1, f2_split2, f2_split3, f2_split4]

        acc_split0 = cv_results['split0_test_Accuracy'][best_f2_index]
        acc_split1 = cv_results['split1_test_Accuracy'][best_f2_index]
        acc_split2 = cv_results['split2_test_Accuracy'][best_f2_index]
        acc_split3 = cv_results['split3_test_Accuracy'][best_f2_index]
        acc_split4 = cv_results['split4_test_Accuracy'][best_f2_index]

        acc_split = [acc_split0, acc_split1, acc_split2, acc_split3, acc_split4]

        # Based on best accuracy
        f2_split0 = calc_f2_score(cv_results['split0_test_Precision'], cv_results['split0_test_Recall'], 2)[best_accuracy_index]
        f2_split1 = calc_f2_score(cv_results['split1_test_Precision'], cv_results['split1_test_Recall'], 2)[best_accuracy_index]
        f2_split2 = calc_f2_score(cv_results['split2_test_Precision'], cv_results['split2_test_Recall'], 2)[best_accuracy_index]
        f2_split3 = calc_f2_score(cv_results['split3_test_Precision'], cv_results['split3_test_Recall'], 2)[best_accuracy_index]
        f2_split4 = calc_f2_score(cv_results['split4_test_Precision'], cv_results['split4_test_Recall'], 2)[best_accuracy_index]

        f2_split_acc = [f2_split0, f2_split1, f2_split2, f2_split3, f2_split4]

        acc_split0 = cv_results['split0_test_Accuracy'][best_accuracy_index]
        acc_split1 = cv_results['split1_test_Accuracy'][best_accuracy_index]
        acc_split2 = cv_results['split2_test_Accuracy'][best_accuracy_index]
        acc_split3 = cv_results['split3_test_Accuracy'][best_accuracy_index]
        acc_split4 = cv_results['split4_test_Accuracy'][best_accuracy_index]

        acc_split_acc = [acc_split0, acc_split1, acc_split2, acc_split3, acc_split4]

        print('Saving scores')
        with open(path + model + '_fold_scores_' + method + '.txt', 'w') as file:
            # Model and method
            file.write('Model: ' + model + ' Method: ' + method + '\n')
            file.write('Fold scores of best parameter set based on F2 scoring\n')

            file.write('Best F2 Parameter\n')
            for key, value in best_params_f2.items():
                file.write('%s: %s\n' % (key, value))
            # Write F2 scores
            file.write('\nF2 Scores\n')
            for item in f2_split:
                file.write('%s\n' % (item))

            # Write accuracy scores
            file.write('\nAccuracy Scores\n')
            for item in acc_split:
                file.write('%s\n' % (item))

            file.write('\n\nFold scores of best parameter set based on accuracy\n')

            file.write('Best Accuracy Parameter\n')
            for key, value in best_params_acc.items():
                file.write('%s: %s\n' % (key, value))
            # Write F2 scores
            file.write('\nF2 Scores\n')
            for item in f2_split_acc:
                file.write('%s\n' % (item))

            # Write accuracy scores
            file.write('\nAccuracy Scores\n')
            for item in acc_split_acc:
                file.write('%s\n' % (item))


print('Wrote parameter information')
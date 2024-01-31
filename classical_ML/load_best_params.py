
# Load best parameters from text file
import pickle

def load_best_params():

    with open('results/best_params_dict.pkl', 'rb') as f:
        best_params = pickle.load(f)

    # print(best_params)

    return best_params


# t = load_best_params()
# params_best = [{'svc__C': 1, 'svc__kernel': 'linear', 'umap__n_components': 3, 'umap__n_neighbors': 5}, {'randomforestclassifier__max_features': 25, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__n_estimators': 10, 'umap__n_components': 3, 'umap__n_neighbors': 5}, {'umap__n_components': 3, 'umap__n_neighbors': 10, 'xgbclassifier__learning_rate': 0.01, 'xgbclassifier__max_depth': 2, 'xgbclassifier__n_estimators': 50}, {'gaussianmixture__init_params': 'k-means++', 'umap__n_components': 3, 'umap__n_neighbors': 5}]

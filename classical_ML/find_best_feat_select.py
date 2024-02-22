import numpy as np
import pickle

def find_best_feat_select(umap_params, umap_scores, ica_params, ica_scores):

    # Param list has 4 dicts that contain param set for each model
    # Score list has 4 F2 scores, one for each model

    # svc_scores = [umap_scores[0], kbest_scores[0], ica_scores[0]]
    # svc_params = [umap_params[0], kbest_params[0], ica_params[0]]

    # rf_scores = [umap_scores[1], kbest_scores[1], ica_scores[1]]
    # rf_params = [umap_params[1], kbest_params[1], ica_params[1]]

    # xg_scores = [umap_scores[2], kbest_scores[2], ica_scores[2]]
    # xg_params = [umap_params[2], kbest_params[2], ica_params[2]]

    # gmm_scores = [umap_scores[3], kbest_scores[3], ica_scores[3]]
    # gmm_params = [umap_params[3], kbest_params[3], ica_params[3]]

    svc_scores = [umap_scores[0], ica_scores[0]]
    svc_params = [umap_params[0], ica_params[0]]

    rf_scores = [umap_scores[1], ica_scores[1]]
    rf_params = [umap_params[1], ica_params[1]]

    xg_scores = [umap_scores[2], ica_scores[2]]
    xg_params = [umap_params[2], ica_params[2]]

    gmm_scores = [umap_scores[3], ica_scores[3]]
    gmm_params = [umap_params[3], ica_params[3]]

    best_svc_idx = np.argmax(svc_scores)
    best_svc_params = svc_params[best_svc_idx]

    best_rf_idx = np.argmax(rf_scores)
    best_rf_params = rf_params[best_rf_idx]

    best_xg_idx = np.argmax(xg_scores)
    best_xg_params = xg_params[best_xg_idx]

    best_gmm_idx = np.argmax(gmm_scores)
    best_gmm_params = gmm_params[best_gmm_idx]

    new_model_params = [best_svc_params, best_rf_params, best_xg_params, best_gmm_params]
    new_model_scores = [np.max(svc_scores), np.max(rf_scores), np.max(xg_scores), np.max(gmm_scores)]

    # Save best params to text file
    file = open('results/best_params.txt','w')
    for item, score in zip(new_model_params, new_model_scores):
        for key, value in item.items():
            file.write('%s: %s\n' % (key, value))
        file.write('F2 Score: %s\n\n' % (score))
    file.close()

    # Save best params to load later
    with open('results/best_params_dict.pkl', 'wb') as f:
        pickle.dump(new_model_params, f)

    # Save all F2 scores to load later
    with open('results/classical_ml_scores.pkl', 'wb') as f:
        pickle.dump(new_model_scores, f)

    return new_model_params, new_model_scores


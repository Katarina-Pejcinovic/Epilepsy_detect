import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_list = ['ica_scores.pkl','kbest_scores.pkl','umap_scores.pkl','rnn_results.pkl']
models = ['ica_scores','kbest_scores','umap_scores','rnn_scores']



for name, result in zip(models, results_list):
      with open('results/'+result, 'rb') as f:
            scores = pickle.load(f)
            globals()[name] = scores


f2_data = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(0,5)],
    'ICA': ica_scores[0][0],
    'KBest': kbest_scores[0][0],
    'UMap': umap_scores[0][0],
    'RNN': rnn_scores[0]
})

plt.clf()

sns.set_palette("viridis")
f2_melted_data = pd.melt(f2_data, id_vars='Fold', value_name='F2 Score')
sns.barplot(x='Fold', y='F2 Score', hue='variable', data=f2_melted_data)
plt.xlabel('Fold')
plt.ylabel('F2 Score')
plt.title('F2 Score Across Folds')
plt.legend(title='Model', fontsize='small')

plt.clf()

precision_data = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(0,5)],
    'ICA': ica_scores[0][1],
    'KBest': kbest_scores[0][1],
    'UMap': umap_scores[0][1],
    'RNN': rnn_scores[1]
})

sns.set_palette("viridis")
precision_melted_data = pd.melt(precision_data, id_vars='Fold', value_name='Precision')
sns.barplot(x='Fold', y='Precision', hue='variable', data=precision_melted_data)
plt.xlabel('Fold')
plt.ylabel('Precision')
plt.title('Precision Across Folds')
plt.legend(title='Model', fontsize='small')


plt.clf()

recall_data = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(0,5)],
    'ICA': ica_scores[0][2],
    'KBest': kbest_scores[0][2],
    'UMap': umap_scores[0][2],
    'RNN': rnn_scores[2]
})

sns.set_palette("viridis")
recall_melted_data = pd.melt(recall_data, id_vars='Fold', value_name='Recall')
sns.barplot(x='Fold', y='Recall', hue='variable', data=recall_melted_data)
plt.xlabel('Fold')
plt.ylabel('Recall')
plt.title('Recall Across Folds')
plt.legend(title='Model', fontsize='small')

plt.clf()

accuracy_data = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(0,5)],
    'ICA': ica_scores[0][3],
    'KBest': kbest_scores[0][3],
    'UMap': umap_scores[0][3],
    'RNN': rnn_scores[3]
})

sns.set_palette("viridis")
accuracy_melted_data = pd.melt(recall_data, id_vars='Fold', value_name='Accuracy')
sns.barplot(x='Fold', y='Accuracy', hue='variable', data=accuracy_melted_data)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy Across Folds')
plt.legend(title='Model', fontsize='small')
plt.show()
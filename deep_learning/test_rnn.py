

#generate  data
train = np.random.rand(150, 34, 437500).astype(np.float32)
test = np.random.rand(50, 34, 437500).astype(np.float32)

#generate labels
train_label = np.random.randint(2, size=100)
test_label = np.random.randint(2, size=50)

#test
preds = rnn_model(train, train_label, test)[0]
preds_proba = rnn_model(train, train_label, test)[1]


#confusion matrix
print(confusion_matrix(test_label,preds))
print(accuracy_score(test_label,preds))
print(precision_score(test_label,preds))
print(recall_score(test_label,preds))

#roc-auc
fpr, tpr, thresholds = roc_curve(test_label, preds_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
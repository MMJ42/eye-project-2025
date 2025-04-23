
from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt  # pip install matplotlib
import numpy as np  # pip install numpy
from numpy import interp
from sklearn.preprocessing import label_binarize
import pandas as pd  # pip install pandas

from torchmetrics import Specificity



true_loc = "File for Reproduction\\true_label_test.csv"

true_label = pd.read_csv(true_loc)


predict_loc = "File for Reproduction\\predict_label_test.csv"

predict_data = pd.read_csv(predict_loc)

predict_label = predict_data.to_numpy().argmax(axis=1)

predict_score = predict_data.to_numpy().max(axis=1)


accuracy = accuracy_score(true_label, predict_label)
print("accuracy: ", accuracy)


precision_macro = precision_score(true_label, predict_label, labels=None,
                            average='macro')  # 'micro', 'macro', 'weighted'
precision_micro = precision_score(true_label, predict_label, labels=None,
                            average='micro')  # 'micro', 'macro', 'weighted'
print("precision-P-macro: ", precision_macro)
print("precision-P-micro: ", precision_micro)


recall_macro = recall_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
recall_micro = recall_score(true_label, predict_label, average='micro')  # 'micro', 'macro', 'weighted'
print("recall-macro: ", recall_macro)
print("recall-micro: ", recall_micro)


f1_macro = f1_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
f1_micro = f1_score(true_label, predict_label, average='micro')  # 'micro', 'macro', 'weighted'
print("F1 Score-macro: ", f1_macro)
print("F1 Score-micro: ", f1_micro)


label_names = ["AMD","CNV","CSR","DME","DR","DRUSEN","MH","NORMAL"]
confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])

plt.matshow(confusion, cmap=plt.cm.Oranges)  # Greens, Blues, Oranges, Reds
plt.colorbar()
for i in range(len(confusion)):
    for j in range(len(confusion)):
        plt.annotate(confusion[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=18)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.xticks(range(len(label_names)), label_names, fontsize=16)
plt.yticks(range(len(label_names)), label_names, fontsize=16)
plt.show()
plt.pause(10)

'''
ROC Curve
'''
n_classes = len(label_names)
# binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])
binarize_predict = label_binarize(true_label, classes=[i for i in range(n_classes)])



predict_score = predict_data.to_numpy()


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(binarize_predict[:, i], [socre_i[i] for socre_i in predict_score])
    roc_auc[i] = auc(fpr[i], tpr[i])

# print("roc_auc = ",roc_auc)

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
plt.pause(10)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(label_names[i], roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()
plt.pause(10)


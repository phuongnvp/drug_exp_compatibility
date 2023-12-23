#%% Import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import requests
df1=pd.read_csv('2D_data.csv')
y = df1.iloc[:,-3].values
X_2D = df1.iloc[:,1:399].values
from sklearn.model_selection import train_test_split
X_train_2D, X_remain, y_train, y_remain = train_test_split(X_2D, y, test_size=.4)
X_val_2D, X_test_2D, y_val, y_test = train_test_split(X_remain, y_remain, test_size=.5)
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE()
X_train_resampled_2D, y_train_resampled = svmsmote.fit_resample(X_train_2D, y_train)
import pickle
with open('2D_model.pkl', 'rb') as f:
    model_2D = pickle.load(f)
y_train_2D = model_2D.predict_proba(X_train_resampled_2D)[:, 1]
#%% mol2vec
df2=pd.read_csv('mol2vec_data.csv')
y = df2.iloc[:,-3].values
X_mol2vec = df2.iloc[:,0:200].values
from sklearn.model_selection import train_test_split
X_train_mol2vec, X_remain, y_train, y_remain = train_test_split(X_mol2vec, y, test_size=.4)
X_val_mol2vec, X_test_mol2vec, y_val, y_test = train_test_split(X_remain, y_remain, test_size=.5)
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE()
X_train_resampled_mol2vec, y_train_resampled = svmsmote.fit_resample(X_train_mol2vec, y_train)
import pickle
with open('mol2vec_model.pkl', 'rb') as f:
    model_mol2vec = pickle.load(f)
y_train_mol2vec = model_mol2vec.predict_proba(X_train_resampled_mol2vec)[:, 1]

#%%
y_pred_train = np.stack((y_train_2D, y_train_mol2vec), axis = 1)

#%% Stack model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(y_pred_train, y_train_resampled)

#%% Stack model - Evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
y_pred_2D = model_2D.predict_proba(X_val_2D)[:,1]
y_pred_mol2vec = model_mol2vec.predict_proba(X_val_mol2vec)[:,1]
y_pred_val = np.stack((y_pred_2D, y_pred_mol2vec), axis = 1)
print(y_pred_val.shape)
#%%
y_pred = lr.predict(y_pred_val)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Precision:", precision)
print("Recall: ", recall)
print('AUC:', roc_auc_score(y_val, y_pred))
print('MCC: ', matthews_corrcoef(y_val, y_pred))
#%% Stack model - Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_val, y_pred, labels=lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
disp.plot()
plt.show()

#%% Stack model - Select threshold based on F1 score
from sklearn import metrics
yhat = lr.predict_proba(np.stack((y_pred_2D, y_pred_mol2vec), axis = 1))
probs = yhat[:,1]
thresholds = np.arange(0, 1, 0.001)
def to_labels(pos_probs, threshold):
 return (pos_probs >= threshold).astype('int')
scores = [f1_score(y_val, to_labels(probs, t)) for t in thresholds]
ix = np.argmax(scores)
threshold1 = thresholds[ix]
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

#%% Stack model - Evaluate new threshold
#y_pred_2 = (stack_model.predict_proba(X_val)[:,1] >= thresholds[ix]).astype(bool)
y_pred_2 = (lr.predict_proba(np.stack((y_pred_2D, y_pred_mol2vec), axis = 1))[:,1] >= threshold1).astype(bool)
accuracy_2 = accuracy_score(y_val, y_pred_2)
f1_2 = f1_score(y_val, y_pred_2)
precision_2 = precision_score(y_val, y_pred_2)
recall_2 = recall_score(y_val, y_pred_2)
print("Accuracy:", accuracy_2)
print("F1-score:", f1_2)
print("Precision:", precision_2)
print("Recall: ", recall_2)
print('AUC:', roc_auc_score(y_val, y_pred_2))
print('MCC: ', matthews_corrcoef(y_val, y_pred_2))

#%% Stack model - Confusion matrix of new model
cm_2 = confusion_matrix(y_val, y_pred_2, labels=lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=lr.classes_)
disp.plot()
plt.show()

#%% Stack model - Select threshold using GHOST algorithm (doi.org/10.1021/acs.jcim.1c00160)
import ghostml
from sklearn import metrics
import numpy as np

def calc_metrics(labels_test, test_probs, threshold):
    scores = [1 if x>=threshold else 0 for x in test_probs]
    auc = metrics.roc_auc_score(labels_test, test_probs)
    kappa = metrics.cohen_kappa_score(labels_test,scores)
    confusion = metrics.confusion_matrix(labels_test,scores, labels=list(set(labels_test)))
    print('thresh: %.2f, kappa: %.3f, AUC test-set: %.3f'%(threshold, kappa, auc))
    print(confusion)
    print(metrics.classification_report(labels_test,scores))
    return 

#%% Calculate threshold
train_probs = lr.predict_proba(y_pred_train)[:,1]
thresholds = np.round(np.arange(0,1,0.001),2)
threshold2 = ghostml.optimize_threshold_from_predictions(y_train_resampled, train_probs, thresholds, ThOpt_metrics = 'Kappa') 
y_pred_3 = (lr.predict_proba(np.stack((y_pred_2D, y_pred_mol2vec), axis = 1))[:,1] >= threshold2).astype(bool)
calc_metrics(y_val, y_pred_3, threshold = threshold2)

#%% Stack model - Evaluate new threshold
accuracy_3 = accuracy_score(y_val, y_pred_3)
f1_3 = f1_score(y_val, y_pred_3)
precision_3 = precision_score(y_val, y_pred_3)
print("Accuracy:", accuracy_3)
print("F1-score:", f1_3)
print("Precision:", precision_3)
print("Recall: ", recall_score(y_val, y_pred_3))
print('AUC:', roc_auc_score(y_val, y_pred_3))
print('MCC: ', matthews_corrcoef(y_val, y_pred_3))

#%% Stack model - Confusion matrix of new model
cm_3 = confusion_matrix(y_val, y_pred_3, labels=lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_3, display_labels=lr.classes_)
disp.plot()
plt.show()

#%% Performance on test set
y_pred_test_2D = model_2D.predict_proba(X_test_2D)[:,1]
y_pred_test_mol2vec = model_mol2vec.predict_proba(X_test_mol2vec)[:,1]
y_pred_4 = (lr.predict_proba(np.stack((y_pred_test_2D, y_pred_test_mol2vec), axis = 1))[:,1] >= thresholds1).astype(bool)
calc_metrics(y_test, y_pred_4, threshold = threshold1)
accuracy_4 = accuracy_score(y_test, y_pred_4)
f1_4 = f1_score(y_test, y_pred_4)
precision_4 = precision_score(y_test, y_pred_4)
recall_4 = recall_score(y_test, y_pred_4)
print("Accuracy:", accuracy_4)
print("F1-score:", f1_4)
print("Precision:", precision_4)
print("Recall: ", recall_4)
print('AUC:', roc_auc_score(y_test, y_pred_4))
print('MCC: ', matthews_corrcoef(y_test, y_pred_4))
cm_4 = confusion_matrix(y_test, y_pred_4, labels=lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_4, display_labels=lr.classes_)
disp.plot()
plt.show()

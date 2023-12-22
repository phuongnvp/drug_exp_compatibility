#%% Import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import requests
df=pd.read_csv('mol2vec_data.csv')
df.columns = ['Pub_CID' + str(i+1) if i<100 else col for i, col in enumerate(df.columns)]
df.columns = ['Pub_Excipient' + str(i+1) if 99<i<200 else col for i, col in enumerate(df.columns)]
df.shape
df["Outcome1"].value_counts()

#%% Define X and y
y = df['Outcome1'].values
X = df.drop(columns=["Outcome1", "API_CID", "Excipient_CID"], axis =1)
print(X.shape)
print(y.shape)

#%% Training set, validation set and test set
from sklearn.model_selection import train_test_split
X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=.4)
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=.5)

#%% Handle imbalanced data
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE()
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train, y_train)
X_train_resampled

#%% Random forest and XGBoost
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
model1 = AdaBoostClassifier(learning_rate = 0.7, n_estimators = 700)
model2 = RandomForestClassifier(n_estimators=100)
model3 = XGBClassifier(max_depth=15, learning_rate=0.9, n_estimators=200)

#%% Stack model
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=10)
stack_model = StackingClassifier(estimators = [('ada', model1), ('rf', model2), ('xgb', model3)], final_estimator = lr)

#%% Stack model - Fit data
class_weights = {0: 0.3, 1: 0.7}
stack_model.fit(X_train_resampled, y_train_resampled, sample_weight=[class_weights[i] for i in y_train_resampled])

#%% Stack model - Evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
y_pred = stack_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Precision:", precision)
print("Recall: ", recall)

#%% Stack model - Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_val, y_pred, labels=stack_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=stack_model.classes_)
disp.plot()
plt.show()

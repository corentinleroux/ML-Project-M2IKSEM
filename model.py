import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
sns.set_style("darkgrid")
import missingno as msno
from IPython.display import display
import warnings
warnings.filterwarnings(action="ignore")

df = pd.read_csv("input/divorce.csv")
display(df)

print(f"Dataset consists of {df.shape[0]} rows and {df.shape[1]} columns")

msno.matrix(df)

df.info()

# Statistical analysis
df.describe().T



# labels
sns.countplot(df.Divorce_Y_N)
plt.show()

X = df.drop("Divorce_Y_N",axis=1)
y = df.Divorce_Y_N

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_pred, y_test)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {acc*100} %")
print(report)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True,cmap="icefire")
plt.show()

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred2 = ada.predict(X_test)
acc2 = accuracy_score(y_pred2, y_test)
report2 = classification_report(y_test, y_pred2)
print(f"Accuracy: {acc2*100} %")
print(report2)
cm2 = confusion_matrix(y_pred2, y_test)
sns.heatmap(cm2, annot=True,cmap="icefire")
plt.show()

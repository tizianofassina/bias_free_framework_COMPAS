import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df_train_classical = pd.read_csv("pre-processed-classical-compas-scores-cleaned-trained.csv")

df_train_alternative = pd.read_csv("pre-processed-alternative-compas-scores-cleaned-trained.csv")

df_test_classical = pd.read_csv("pre-processed-classical-compas-scores-cleaned-test.csv")

df_test_alternative = pd.read_csv("pre-processed-alternative-compas-scores-cleaned-test.csv")

df_train_not_processed = pd.read_csv("compas-scores-cleaned-train.csv")
df_test_not_processed = pd.read_csv("compas-scores-cleaned-test.csv")

# I compare the accuracy of the two processing procedures.

X_train_classical = df_train_classical.iloc[:, :-1]
y_train_classical = df_train_classical.iloc[:, -1]
X_train_alternative = df_train_alternative.iloc[:, :-1]
y_train_alternative = df_train_alternative.iloc[:, -1]

X_test_classical = df_test_classical.iloc[:, :-1]
X_test_alternative = df_test_alternative.iloc[:, :-1]

Y_test = df_test_not_processed.iloc[:,-1]

model_classical = LogisticRegression(max_iter=1000)
model_classical.fit(X_train_classical , y_train_classical)

model_alternative = LogisticRegression(max_iter=1000)
model_alternative.fit(X_train_alternative , y_train_alternative)

y_pred_classical = model_classical.predict(X_test_classical)
y_pred_alternative = model_alternative.predict(X_test_alternative)

accuracy_classical = accuracy_score(Y_test, y_pred_classical)
accuracy_alternative = accuracy_score(Y_test, y_pred_alternative)

print("Accuracy of classical method : ", accuracy_classical)
print("Accuracy of alternative method : ", accuracy_alternative
      )



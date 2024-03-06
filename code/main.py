from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

dataset = pd.read_csv('form_answers.csv')

dataset.drop(columns=["date"], inplace=True)

categorical_columns = ['sex', 'age', 'education', 'salary', 'fav']
dataset = pd.get_dummies(dataset, columns=categorical_columns)

X = dataset.drop('output', axis=1)
y = dataset['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
cv_scores = cross_val_score(knn_model, X, y, cv=5)

k_folds = 5
data_length = len(X)
fold_size = data_length // k_folds

accuracy_scores = []

for i in range(k_folds):
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < k_folds - 1 else data_length

    X_test_fold = X.iloc[start_idx:end_idx, :]
    y_test_fold = y.iloc[start_idx:end_idx]

    X_train_fold = pd.concat([X.iloc[0:start_idx, :], X.iloc[end_idx:data_length, :]])
    y_train_fold = pd.concat([y.iloc[0:start_idx], y.iloc[end_idx:data_length]])

    knn_model.fit(X_train_fold, y_train_fold)

    y_pred_fold = knn_model.predict(X_test_fold)

    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
    accuracy_scores.append(accuracy_fold)

print(f'Cross-Validation Doğruluk Skorları: {accuracy_scores}')
print(f'Ortalama Doğruluk Skoru: {sum(accuracy_scores) / k_folds}')

print(f'Model Doğruluğu: {accuracy}')
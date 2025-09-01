import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

data.dropna(inplace=True)

X = data.drop('Survived', axis=1)
y = data['Survived']
X = X.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, conf_matrix

log_reg_metrics = evaluate_model(log_reg, X_test, y_test)
knn_metrics = evaluate_model(knn, X_test, y_test)

print("Logistic Regression Metrics:", log_reg_metrics)
print("KNN Metrics:", knn_metrics)

def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Did Not Survive', 'Survived'], 
                yticklabels=['Did Not Survive', 'Survived'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

plot_confusion_matrix(log_reg_metrics[4], title='Logistic Regression Confusion Matrix')
plot_confusion_matrix(knn_metrics[4], title='KNN Confusion Matrix')

classifier_names = ['Logistic Regression', 'KNN']
accuracies = [log_reg_metrics[0], knn_metrics[0]]

plt.bar(classifier_names, accuracies, color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Classifier Accuracies')
plt.show()

print("Interpretation of Logistic Regression:")
print(f"Accuracy: {log_reg_metrics[0]}, Precision: {log_reg_metrics[1]}, Recall: {log_reg_metrics[2]}, F1-score: {log_reg_metrics[3]}")

print("Interpretation of KNN:")
print(f"Accuracy: {knn_metrics[0]}, Precision: {knn_metrics[1]}, Recall: {knn_metrics[2]}, F1-score: {knn_metrics[3]}")

best_classifier = "Logistic Regression" if log_reg_metrics[0] > knn_metrics[0] else "KNN"
print(f"The best classifier overall is: {best_classifier}")
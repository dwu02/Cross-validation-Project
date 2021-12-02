import time as t
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

def run_trials(id,params,N,X_train,y_train,X_test,y_test):
    train_runtime = []
    pred_runtime = []
    accuracy = []
    if id == 'knn':
        model = KNeighborsClassifier(**params)
    elif id == 'tree':
        model = DecisionTreeClassifier(**params)
    elif id == 'rfc': 
        model = RandomForestClassifier(**params)
    elif id == 'svm_poly' or id == 'svm_rbf':
        model = SVC(**params)
    elif id == 'deep_sigmoid' or id == 'deep_relu':
        model = MLPClassifier(**params)
    for trial in range(N):
        checkpoint_1 = t.time()
        model.fit(X_train,y_train)
        checkpoint_2 = t.time()
        yhat = model.predict(X_test)
        checkpoint_3 = t.time()
        acc = accuracy_score(y_test, yhat)
        train_runtime.append(checkpoint_2-checkpoint_1)
        pred_runtime.append(checkpoint_3-checkpoint_2)
        accuracy.append(acc)
    avg_train_runtime = np.mean(train_runtime)
    std_train_runtime = np.std(train_runtime)
    avg_pred_runtime = np.mean(pred_runtime)
    std_pred_runtime = np.std(pred_runtime)
    avg_acc = np.mean(accuracy)
    std_acc = np.std(accuracy)
    print(f"""
Trials: {N}
Average Training Runtime: {avg_train_runtime}
Standard Deviation Training Runtime: {std_train_runtime}
Average Prediction Runtime: {avg_pred_runtime}
Standard Deviation Prediction Runtime: {std_pred_runtime}
Average Accuracy: {avg_acc}
Standard Deviation of Accuracy: {std_acc}
    """)

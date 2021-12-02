from process_data import pipeline
import time as t
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from algorithm_runner import deep_relu, deep_sigmoid, knn_run,rfc_run, svm_poly, svm_rbf, tree_run
from trials import run_trials

FILE_PATH = 'data/breast-cancer-wisconsin.data'
X, y, folds = pipeline(FILE_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
if sys.argv[1] == 'knn':
    tuned_params = knn_run(X,y,folds)
    # tuned_model = KNeighborsClassifier(**tuned_params)
    print('KNN Classifier')

elif sys.argv[1] == 'tree':
    tuned_params = tree_run(X,y,folds)
    # tuned_model = DecisionTreeClassifier(**tuned_params)
    print('Decision Tree Classifier')

elif sys.argv[1] == 'rfc':
    tuned_params = rfc_run(X,y,folds)
    # tuned_model = RandomForestClassifier(**tuned_params)
    print('Random Forest Classifier')

elif sys.argv[1] == 'svm_poly':
    tuned_params = svm_poly(X,y,folds)
    # tuned_model = SVC(**tuned_params)
    print('SVM with Polynomial Kernel')

elif sys.argv[1] == 'svm_rbf':
    tuned_params = svm_rbf(X,y,folds)
    # tuned_model = SVC(**tuned_params)
    print('SVM with RBF Kernel')

elif sys.argv[1] == 'deep_sigmoid':
    tuned_params = deep_sigmoid(X,y,folds)
    # tuned_model = MLPClassifier(**tuned_params)
    print('Deep neural network with sigmoid activation')

elif sys.argv[1] == 'deep_relu':
    tuned_params = deep_relu(X,y,folds)
    # tuned_model = MLPClassifier(**tuned_params)
    print('Deep neural network with RELU activation')

run_trials(sys.argv[1],tuned_params,1000,X_train,y_train,X_test,y_test)

# checkpoint_1 = t.time()
# tuned_model.fit(X_train,y_train)
# checkpoint_2 = t.time()
# yhat = tuned_model.predict(X_test)
# checkpoint_3 = t.time()
# acc = accuracy_score(y_test, yhat)
# print(list(y_test))
# print(yhat)
# print(f'Training Runtime: {checkpoint_2-checkpoint_1} seconds')
# print(f'Prediction Runtime: {checkpoint_3-checkpoint_2} seconds')
# print(f'Final Validation Set Accuracy: {acc}')
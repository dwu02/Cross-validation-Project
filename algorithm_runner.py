# § Brief description of the classifier and its general advantages/disadvantages 
# § Figure: 
# Graph the cross validation results over the range of
# hyperparameter values you tested
# • Bonus points for additional hyperparameters tuned
# § Any additional details needed to replicate your results


# graphs
# mean train score across ten splits
# mean test score
# mean score time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV
from grapher import deep_relu_display, deep_sigmoid_display, knn_display, rfc_display, svm_poly_display, svm_rbf_display, tree_display
import pandas as pd

def knn_run(X,y,folds):
    knn_params = {'n_neighbors' : [2,3,5,10,15],
                'p': [1,2,3]}
    knn_tuner = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = knn_params,cv=folds, return_train_score=True,scoring="accuracy",verbose=0)
    knn_tuner.fit(X,y)
    data = {'Neighbors':[2,2,2,3,3,3,5,5,5,10,10,10,15,15,15],
        'Distance Metric':[1,2,3,1,2,3,1,2,3,1,2,3,1,2,3],
        'Mean Train Score':[i for i in knn_tuner.cv_results_['mean_train_score']],
        'Mean Test Score':[i for i in knn_tuner.cv_results_['mean_test_score']],
        'Mean Score Time':[i for i in knn_tuner.cv_results_['mean_score_time']]
       }
    df = pd.DataFrame(data)
    n_1 = df.loc[df['Distance Metric'] == 1]
    n_2 = df.loc[df['Distance Metric'] == 2]
    n_3 = df.loc[df['Distance Metric'] == 3]
    # knn_display(n_1,n_2,n_3)
    # ratio = knn_tuner.best_score_/np.mean(data['Mean Score Time'])
    # print(f'Best Score on Validation Set: {knn_tuner.best_score_}')
    # print(f'Best Parameters: {knn_tuner.best_params_}')
    # print(f'Ratio between Score and Scoring Time for 150 fits {ratio}')
    return knn_tuner.best_params_

def tree_run(X,y,folds):
    tree_params = {'max_depth': [2, 3, 5],
                   'min_samples_leaf': [5, 10, 20, 50, 100]}
    tree_tuner = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = tree_params,cv=folds, return_train_score=True,scoring="accuracy",verbose=0)
    tree_tuner.fit(X,y)
    data = {'max_depth':[2,2,2,2,2,3,3,3,3,3,5,5,5,5,5],
        'min_samples_leaf':[5, 10, 20, 50, 100, 5, 10, 20, 50, 100, 5, 10, 20, 50, 100],
        'Mean Train Score':[i for i in tree_tuner.cv_results_['mean_train_score']],
        'Mean Test Score':[i for i in tree_tuner.cv_results_['mean_test_score']],
        'Mean Score Time':[i for i in tree_tuner.cv_results_['mean_score_time']]
       }
    df = pd.DataFrame(data)
    n_1 = df.loc[df['max_depth'] == 2]
    n_2 = df.loc[df['max_depth'] == 3]
    n_3 = df.loc[df['max_depth'] == 5]
    # tree_display(n_1,n_2,n_3)
    # ratio = tree_tuner.best_score_/np.mean(data['Mean Score Time'])
    # print(f'Best Score on Validation Set: {tree_tuner.best_score_}')
    # print(f'Best Parameters: {tree_tuner.best_params_}')
    # print(f'Ratio between Score and Scoring Time for 150 fits {ratio}')
    return tree_tuner.best_params_

def rfc_run(X,y,folds):
    rfc_params = {'max_depth': [2, 3, 5],
                  'n_estimators' : [4,6,10,20,30]}
    rfc_tuner = GridSearchCV(estimator = RandomForestClassifier(), param_grid = rfc_params,cv=folds, return_train_score=True,scoring="accuracy",verbose=0)
    rfc_tuner.fit(X,y)
    data = {'max_depth':[2,2,2,2,2,3,3,3,3,3,5,5,5,5,5],
        'n_estimators':[4,6,10,20,30,4,6,10,20,30,4,6,10,20,30],
        'Mean Train Score':[i for i in rfc_tuner.cv_results_['mean_train_score']],
        'Mean Test Score':[i for i in rfc_tuner.cv_results_['mean_test_score']],
        'Mean Score Time':[i for i in rfc_tuner.cv_results_['mean_score_time']]
       }
    df = pd.DataFrame(data)
    n_1 = df.loc[df['max_depth'] == 2]
    n_2 = df.loc[df['max_depth'] == 3]
    n_3 = df.loc[df['max_depth'] == 5]
    # rfc_display(n_1,n_2,n_3)
    # ratio = rfc_tuner.best_score_/np.mean(data['Mean Score Time'])
    # print(f'Best Score on Validation Set: {rfc_tuner.best_score_}')
    # print(f'Best Parameters: {rfc_tuner.best_params_}')
    # print(f'Ratio between Score and Scoring Time for 150 fits {ratio}')
    return rfc_tuner.best_params_

def svm_poly(X,y,folds):
    svm_poly_params = {'kernel' : ['poly'],
                'degree': [2,3,4],
                'C': [0.05,0.1,1,5,10]}
    svm_poly_tuner = GridSearchCV(estimator = SVC(), param_grid = svm_poly_params,cv=folds, return_train_score=True,scoring="accuracy",verbose=0)
    svm_poly_tuner.fit(X,y)
    data = {'degree':[2,3,4,2,3,4,2,3,4,2,3,4,2,3,4],
        'Regularization Parameter':[0.05,0.05,0.05,0.1,0.1,0.1,1,1,1,5,5,5,10,10,10],
        'Mean Train Score':[i for i in svm_poly_tuner.cv_results_['mean_train_score']],
        'Mean Test Score':[i for i in svm_poly_tuner.cv_results_['mean_test_score']],
        'Mean Score Time':[i for i in svm_poly_tuner.cv_results_['mean_score_time']]
       }
    df = pd.DataFrame(data)
    n_1 = df.loc[df['degree'] == 2]
    n_2 = df.loc[df['degree'] == 3]
    n_3 = df.loc[df['degree'] == 4]
    # svm_poly_display(n_1,n_2,n_3)
    # ratio = svm_poly_tuner.best_score_/np.mean(data['Mean Score Time'])
    # print(f'Best Score on Validation Set: {svm_poly_tuner.best_score_}')
    # print(f'Best Parameters: {svm_poly_tuner.best_params_}')
    # print(f'Ratio between Score and Scoring Time for 150 fits {ratio}')
    return svm_poly_tuner.best_params_

def svm_rbf(X,y,folds):
    svm_rbf_params = {'kernel' : ['rbf'],
                      'C': [0.05,0.1,1,5,10],
                      'gamma': [1,0.1,0.01]}
    svm_rbf_tuner = GridSearchCV(estimator = SVC(), param_grid = svm_rbf_params,cv=folds, return_train_score=True,scoring="accuracy",verbose=0)
    svm_rbf_tuner.fit(X,y)
    data = {'gamma':[1,0.1,0.01,1,0.1,0.01,1,0.1,0.01,1,0.1,0.01,1,0.1,0.01],
        'Regularization Parameter':[0.05,0.05,0.05,0.1,0.1,0.1,1,1,1,5,5,5,10,10,10],
        'Mean Train Score':[i for i in svm_rbf_tuner.cv_results_['mean_train_score']],
        'Mean Test Score':[i for i in svm_rbf_tuner.cv_results_['mean_test_score']],
        'Mean Score Time':[i for i in svm_rbf_tuner.cv_results_['mean_score_time']]
       }
    df = pd.DataFrame(data)
    n_1 = df.loc[df['gamma'] == 1]
    n_2 = df.loc[df['gamma'] == 0.1]
    n_3 = df.loc[df['gamma'] == 0.01]
    # svm_rbf_display(n_1,n_2,n_3)
    # ratio = svm_rbf_tuner.best_score_/np.mean(data['Mean Score Time'])
    # print(f'Best Score on Validation Set: {svm_rbf_tuner.best_score_}')
    # print(f'Best Parameters: {svm_rbf_tuner.best_params_}')
    # print(f'Ratio between Score and Scoring Time for 150 fits {ratio}')
    return svm_rbf_tuner.best_params_

def deep_sigmoid(X,y,folds):
    deep_sigmoid_params = {'activation' : ['logistic'],
                           'alpha' : [1,0.1,0.01],
                           'hidden_layer_sizes': [(6,2),(6,3),(6,4),(6,5),(6,6)]}
    deep_sigmoid_tuner = GridSearchCV(estimator = MLPClassifier(), param_grid = deep_sigmoid_params,cv=folds, return_train_score=True,scoring="accuracy",verbose=0)
    deep_sigmoid_tuner.fit(X,y)
    data = {'Regularization Parameter':[1,1,1,1,1,0.1,0.1,0.1,0.1,0.1,0.01,0.01,0.01,0.01,0.01],
        'hidden layers':[2,3,4,5,6,2,3,4,5,6,2,3,4,5,6],
        'Mean Train Score':[i for i in deep_sigmoid_tuner.cv_results_['mean_train_score']],
        'Mean Test Score':[i for i in deep_sigmoid_tuner.cv_results_['mean_test_score']],
        'Mean Score Time':[i for i in deep_sigmoid_tuner.cv_results_['mean_score_time']]
    }
    df = pd.DataFrame(data)
    n_1 = df.loc[df['Regularization Parameter'] == 1]
    n_2 = df.loc[df['Regularization Parameter'] == 0.1]
    n_3 = df.loc[df['Regularization Parameter'] == 0.01]
    # deep_sigmoid_display(n_1,n_2,n_3)
    # ratio = deep_sigmoid_tuner.best_score_/np.mean(data['Mean Score Time'])
    # print(f'Best Score on Validation Set: {deep_sigmoid_tuner.best_score_}')
    # print(f'Best Parameters: {deep_sigmoid_tuner.best_params_}')
    # print(f'Ratio between Score and Scoring Time for 150 fits {ratio}')
    return deep_sigmoid_tuner.best_params_

def deep_relu(X,y,folds):
    deep_relu_params = {'activation' : ['relu'],
                           'alpha' : [1,0.1,0.01],
                           'hidden_layer_sizes': [(6,2),(6,3),(6,4),(6,5),(6,6)]}
    deep_relu_tuner = GridSearchCV(estimator = MLPClassifier(), param_grid = deep_relu_params,cv=folds, return_train_score=True,scoring="accuracy",verbose=0)
    deep_relu_tuner.fit(X,y)
    data = {'Regularization Parameter':[1,1,1,1,1,0.1,0.1,0.1,0.1,0.1,0.01,0.01,0.01,0.01,0.01],
        'hidden layers':[2,3,4,5,6,2,3,4,5,6,2,3,4,5,6],
        'Mean Train Score':[i for i in deep_relu_tuner.cv_results_['mean_train_score']],
        'Mean Test Score':[i for i in deep_relu_tuner.cv_results_['mean_test_score']],
        'Mean Score Time':[i for i in deep_relu_tuner.cv_results_['mean_score_time']]
    }
    df = pd.DataFrame(data)
    n_1 = df.loc[df['Regularization Parameter'] == 1]
    n_2 = df.loc[df['Regularization Parameter'] == 0.1]
    n_3 = df.loc[df['Regularization Parameter'] == 0.01]
    # deep_relu_display(n_1,n_2,n_3)
    # ratio = deep_relu_tuner.best_score_/np.mean(data['Mean Score Time'])
    # print(f'Best Score on Validation Set: {deep_relu_tuner.best_score_}')
    # print(f'Best Parameters: {deep_relu_tuner.best_params_}')
    # print(f'Ratio between Score and Scoring Time for 150 fits {ratio}')
    return deep_relu_tuner.best_params_
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def knn_display(n_1,n_2,n_3):
    fig1, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (20,5))
    fig1.patch.set_facecolor('lightsteelblue')
    fig1.suptitle("Mean Scores for KNN")
    n_1.plot(kind= "line", x = "Neighbors", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "scatter", x = "Neighbors", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "line", x = "Neighbors", y = "Mean Test Score", color="g", ax=ax0)
    n_1.plot(kind= "scatter", x = "Neighbors", y = "Mean Test Score", color="g", ax=ax0)
    ax0.set(title = "Manhattan Distance", xlabel="Neighbors", ylabel= "Accuracy")
    n_2.plot(kind= "line", x = "Neighbors", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "scatter", x = "Neighbors", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "line", x = "Neighbors", y = "Mean Test Score", color="g", ax=ax1)
    n_2.plot(kind= "scatter", x = "Neighbors", y = "Mean Test Score", color="g", ax=ax1)
    ax1.set(title = "Euclidean Distance", xlabel="Neighbors", ylabel= "Accuracy")
    n_3.plot(kind= "line", x = "Neighbors", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "scatter", x = "Neighbors", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "line", x = "Neighbors", y = "Mean Test Score", color="g", ax=ax2)
    n_3.plot(kind= "scatter", x = "Neighbors", y = "Mean Test Score", color="g", ax=ax2)
    ax2.set(title = "Minkowski Distance", xlabel="Neighbors", ylabel= "Accuracy")
    plt.show()

# def knn_display_time(n_1,n_2,n_3):
#     fig1, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (20,5))
#     fig1.suptitle("Mean Score Time for KNN")
#     n_1.plot(kind= "line", x = "Neighbors", y = "Mean Score Time", color="r", ax=ax0)
#     ax0.set(title = "Manhattan Distance", xlabel="Neighbors", ylabel= "Accuracy")
#     ax0.set_xlim([0,21])
#     n_2.plot(kind= "line", x = "Neighbors", y = "Mean Score Time", color="g", ax=ax1)
#     ax1.set(title = "Euclidean Distance", xlabel="Neighbors", ylabel= "Accuracy")
#     ax1.set_xlim([0,21])
#     n_3.plot(kind= "line", x = "Neighbors", y = "Mean Score Time", color="b", ax=ax2)
#     ax2.set(title = "Minkowski Distance", xlabel="Neighbors", ylabel= "Accuracy")
#     ax2.set_xlim([0,21])
#     plt.show()

def tree_display(n_1,n_2,n_3):
    fig1, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (20,5))
    fig1.patch.set_facecolor('lightsteelblue')
    fig1.suptitle("Mean Scores for Decision Tree")
    n_1.plot(kind= "line", x = "min_samples_leaf", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "scatter", x = "min_samples_leaf", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "line", x = "min_samples_leaf", y = "Mean Test Score", color="g", ax=ax0)
    n_1.plot(kind= "scatter", x = "min_samples_leaf", y = "Mean Test Score", color="g", ax=ax0)
    ax0.set(title = "max_depth = 2", xlabel="min_samples_leaf", ylabel= "Accuracy")
    n_2.plot(kind= "line", x = "min_samples_leaf", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "scatter", x = "min_samples_leaf", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "line", x = "min_samples_leaf", y = "Mean Test Score", color="g", ax=ax1)
    n_2.plot(kind= "scatter", x = "min_samples_leaf", y = "Mean Test Score", color="g", ax=ax1)
    ax1.set(title = "max_depth = 3", xlabel="min_samples_leaf", ylabel= "Accuracy")
    n_3.plot(kind= "line", x = "min_samples_leaf", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "scatter", x = "min_samples_leaf", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "line", x = "min_samples_leaf", y = "Mean Test Score", color="g", ax=ax2)
    n_3.plot(kind= "scatter", x = "min_samples_leaf", y = "Mean Test Score", color="g", ax=ax2)
    ax2.set(title = "max_depth = 5", xlabel="min_samples_leaf", ylabel= "Accuracy")
    plt.show()

def rfc_display(n_1,n_2,n_3):
    fig1, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (20,5))
    fig1.patch.set_facecolor('lightsteelblue')
    fig1.suptitle("Mean Scores for Random Forest")
    n_1.plot(kind= "line", x = "n_estimators", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "scatter", x = "n_estimators", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "line", x = "n_estimators", y = "Mean Test Score", color="g", ax=ax0)
    n_1.plot(kind= "scatter", x = "n_estimators", y = "Mean Test Score", color="g", ax=ax0)
    ax0.set(title = "max_depth = 2", xlabel="n_estimators", ylabel= "Accuracy")
    n_2.plot(kind= "line", x = "n_estimators", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "scatter", x = "n_estimators", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "line", x = "n_estimators", y = "Mean Test Score", color="g", ax=ax1)
    n_2.plot(kind= "scatter", x = "n_estimators", y = "Mean Test Score", color="g", ax=ax1)
    ax1.set(title = "max_depth = 3", xlabel="n_estimators", ylabel= "Accuracy")
    n_3.plot(kind= "line", x = "n_estimators", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "scatter", x = "n_estimators", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "line", x = "n_estimators", y = "Mean Test Score", color="g", ax=ax2)
    n_3.plot(kind= "scatter", x = "n_estimators", y = "Mean Test Score", color="g", ax=ax2)
    ax2.set(title = "max_depth = 5", xlabel="n_estimators", ylabel= "Accuracy")
    plt.show()

def svm_poly_display(n_1,n_2,n_3):
    fig1, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (20,5))
    fig1.patch.set_facecolor('lightsteelblue')
    fig1.suptitle("Mean Scores for SVM with Polynomial Kernel")
    n_1.plot(kind= "line", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "line", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax0)
    n_1.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax0)
    ax0.set(title = "degree = 2", xlabel="Regularization Parameter", ylabel= "Accuracy")
    n_2.plot(kind= "line", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "line", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax1)
    n_2.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax1)
    ax1.set(title = "degree = 3", xlabel="Regularization Parameter", ylabel= "Accuracy")
    n_3.plot(kind= "line", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "line", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax2)
    n_3.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax2)
    ax2.set(title = "degree = 4", xlabel="Regularization Parameter", ylabel= "Accuracy")
    plt.show()

def svm_rbf_display(n_1,n_2,n_3):
    fig1, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (20,5))
    fig1.patch.set_facecolor('lightsteelblue')
    fig1.suptitle("Mean Scores for SVM with RBF Kernel")
    n_1.plot(kind= "line", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "line", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax0)
    n_1.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax0)
    ax0.set(title = "gamma = 1", xlabel="Regularization Parameter", ylabel= "Accuracy")
    n_2.plot(kind= "line", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "line", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax1)
    n_2.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax1)
    ax1.set(title = "gamma = 0.1", xlabel="Regularization Parameter", ylabel= "Accuracy")
    n_3.plot(kind= "line", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "line", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax2)
    n_3.plot(kind= "scatter", x = "Regularization Parameter", y = "Mean Test Score", color="g", ax=ax2)
    ax2.set(title = "gamma = 0.01", xlabel="Regularization Parameter", ylabel= "Accuracy")
    plt.show()

def deep_sigmoid_display(n_1,n_2,n_3):
    fig1, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (20,5))
    fig1.patch.set_facecolor('lightsteelblue')
    fig1.suptitle("Mean Scores for Deep Neural Network with Sigmoid Activation")
    n_1.plot(kind= "line", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "scatter", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "line", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax0)
    n_1.plot(kind= "scatter", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax0)
    ax0.set(title = "Regularization Parameter = 1", xlabel="hidden layers", ylabel= "Accuracy")
    n_2.plot(kind= "line", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "scatter", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "line", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax1)
    n_2.plot(kind= "scatter", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax1)
    ax1.set(title = "Regularization Parameter = 0.1", xlabel="hidden layers", ylabel= "Accuracy")
    n_3.plot(kind= "line", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "scatter", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "line", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax2)
    n_3.plot(kind= "scatter", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax2)
    ax2.set(title = "Regularization Parameter = 0.01", xlabel="hidden layers", ylabel= "Accuracy")
    plt.show()

def deep_relu_display(n_1,n_2,n_3):
    fig1, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (20,5))
    fig1.patch.set_facecolor('lightsteelblue')
    fig1.suptitle("Mean Scores for Deep Neural Network with ReLU Activation")
    n_1.plot(kind= "line", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "scatter", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax0)
    n_1.plot(kind= "line", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax0)
    n_1.plot(kind= "scatter", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax0)
    ax0.set(title = "Regularization Parameter = 1", xlabel="hidden layers", ylabel= "Accuracy")
    n_2.plot(kind= "line", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "scatter", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax1)
    n_2.plot(kind= "line", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax1)
    n_2.plot(kind= "scatter", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax1)
    ax1.set(title = "Regularization Parameter = 0.1", xlabel="hidden layers", ylabel= "Accuracy")
    n_3.plot(kind= "line", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "scatter", x = "hidden layers", y = "Mean Train Score", color="r", ax=ax2)
    n_3.plot(kind= "line", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax2)
    n_3.plot(kind= "scatter", x = "hidden layers", y = "Mean Test Score", color="g", ax=ax2)
    ax2.set(title = "Regularization Parameter = 0.01", xlabel="hidden layers", ylabel= "Accuracy")
    plt.show()
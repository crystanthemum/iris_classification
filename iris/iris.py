import sys, scipy, numpy, pandas, sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the iris dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape - find out how many instances (rows) and how many attributes (col) the data contains
# with the shape property
print(dataset.shape)

#head = peek at the dataset
print(dataset.head(20))

# summary of each attribute : count, mean, min, max values
print(dataset.describe())

# number of instances(row) that belong to each class aka class distribution
print(dataset.groupby('class').size())

# Datavisualization - univariate plots aka plots of each individual variable
# numeric variables means box and whiskers plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey = False)
plt.show()

# Histograms of each input variable to get an idea of the distribution
dataset.hist()
plt.show()

# Multivariate Plots : Scatter plot matrix
# Diagonal grouping of some pair of attributes suggests a high correlation and a predictable relationship
scatter_matrix(dataset)
plt.show()

# Evaluate some algorithms
# 1. Separate out a validation dataset
# 2. Set up the test harness to use 10-fold cross validation
# 3. Build 5 different models to predict species from flower measurements
# 4. Select the best model

# 1. Create a validation dataset - to know if our created models are good
# Need to use statistical methods to estimate the accuracy of the models that we create on unseen data
# Also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual
# unseen data

# First split the dataset into two, 80% used to train models
# 20% as the testing set

# Training data in the X_train
# Preparing models in Y_train
# X and Y_validation sets to be used later

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test harness
# Test options and evaluation metric - splitting dataset into 10 parts, train on 9, test on 1 and repeat for all
# combinations of train-test splits

# accuracy is a metric used to evaluate models.
# scoring variable will be used when we build and evaluate each model

seed = 7
scoring = 'accuracy'

# Building and evaluating 6 models based on 6 algorithms
# Spot check algorithms

# array of models
models= []
# Logistic Regression (LR) algorithm
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

# Linear Discriminat Analysis (LDA) algorithm
models.append(('LDA', LinearDiscriminantAnalysis()))

# K-Nearest Neighbours (KNN) algorithm
models.append(('KNN', KNeighborsClassifier()))

# Classification and Regression Trees (CART) algorithm
models.append(('CART', DecisionTreeClassifier()))

# Gaussian Naive Bayes (NB) algorithm
models.append(('NB', GaussianNB()))

# Support Vector Machines (SVM) algorithm
models.append(('SVM', SVC(gamma='auto')))

#array of results and names
results = []
names = []
print("\nComparing raw results of all the models")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring= scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s:  mean : %f standard deviation : (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# Creating a plot of the model evaluation results and compare the spread
# and the mean accuracy of each model
# There is a population of accuracy measures for each algorithm because
# each algorithm was evaluated 10 times (10 fold cross validation)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# If the box and whisker plots are squashed at the top of the range, that means
# many samples are achieving 100% accuracy

# Making Predictions on the validation dataset and check its accuracy
# Validation set is needed just in case the training data was overfitted or a data leak
print("\n")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("Accuracy score : ",accuracy_score(Y_validation, predictions),"\n")

# Confusion matrix provides an indication of the three errors made.
print(confusion_matrix(Y_validation, predictions))

# Classification report provides a breakdown of each class by precision, recall
# f1-score and support showing excellent results
print(classification_report(Y_validation, predictions))


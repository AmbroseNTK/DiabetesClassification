# Create Classification Pipeline 
# Import libraries
import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import data
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score,f1_score,recall_score
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LeakyReLU
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from read_nn_model import VisualizeTrainingSteps
from sklearn.model_selection import cross_validate, KFold
import joblib
import shutil
import os
from classification_model import ClassificationModel
from sklearn.inspection import permutation_importance

# read args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_file_name", help="file name of the dataset", type=str, default="dataset.csv")
parser.add_argument("--test_size", help="test size", type=float, default=0.2)
parser.add_argument("--save_dir", help="directory to save the models", type=str, default="models")
parser.add_argument("--auto_select_features", help="auto select n features", type=int, default=0)
parser.add_argument("--remove_outliers", help="remove outliers", type=bool, default=True)
parser.add_argument("--normalize", help="normalize data", type=bool, default=False)
parser.add_argument("--save_normalized_data", help="save normalized data", type=bool, default=False)
parser.add_argument("--calc_weights", help="calculate weights", type=bool, default=True)

args = parser.parse_args()
save_dir = args.save_dir

# create dir if not existed or delete if existed

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

os.makedirs(save_dir)
os.makedirs(save_dir + "/models")
os.makedirs(save_dir + "/plots")

def load_data(file_name, test_size=0.2):
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:, 0:len(dataset.columns)-1].values
    y = dataset.iloc[:, len(dataset.columns)-1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    # Save test data to csv
    test_data = pd.DataFrame(X_test)
    test_data['Outcome'] = y_test
    test_data.to_csv(save_dir+"/test_data.csv")
    weights = {}
    if args.calc_weights:
        weights = class_weight.compute_class_weight('balanced',
                                            classes = np.unique(y_train),
                                            y = y_train)
        weights = {0:weights[0], 1:weights[1]}
    return dataset, X_train, X_test, y_train, y_test, weights

def build_neural_network_2_layers(input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(26, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # loss with weighted binary cross entropy

    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_neural_network_leaky_relu(input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(26, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation=LeakyReLU()),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # loss with weighted binary cross entropy

    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def custom_fit_nn(model, X_train, y_train, weights):
    callbacks = [VisualizeTrainingSteps(model,save_dir+'/plots',model.name,10)]
    model.fit(X_train, y_train, epochs=400, class_weight=weights, batch_size=4, callbacks=callbacks)

def custom_fit_xgboost(model, X_train, y_train, weights):
    model.fit(X_train, y_train)

def custom_evaluate_nn(model, X_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    return y_pred


def build_classifier(input_size):
    models = [ClassificationModel('XGBoost', XGBClassifier(use_label_encoder=False, n_estimators=100, learning_rate=0.1, max_depth=5), custom_fit_xgboost),
              ClassificationModel('NaiveBayes', GaussianNB()),
              ClassificationModel('RandomForest', RandomForestClassifier() ),
              ClassificationModel('LogisticRegression', LogisticRegression()),
              ClassificationModel('KNN', KNeighborsClassifier()),
              ClassificationModel('DecisionTree', DecisionTreeClassifier(max_depth=5)),
              ClassificationModel('AdaBoost', AdaBoostClassifier()),
              ClassificationModel('SVM', SVC(probability=True)),
              ClassificationModel('NeuralNetwork2Layers', build_neural_network_2_layers(input_size),custom_fit_nn, custom_evaluate_nn, skip=True),
            #   ClassificationModel('NeuralNetworkLeakyReLU', build_neural_network_leaky_relu(input_size), custom_fit_nn, custom_evaluate_nn)
            ]

    return models

def train_models(models, X_train, y_train, weights):
    for model in models:
        if model.skip:
            continue
        if model.custom_fit_fn is not None:
            model.custom_fit_fn(model.model, X_train, y_train, weights)
        else:
            model.model.fit(X_train, y_train)
        # Save model
        try:
            model.model.save(save_dir + "/" + model.model_name + ".h5")
        except:
            joblib.dump(model.model, save_dir + "/" + model.model_name + ".pkl")

def evaluate_models_with_k_folds(model, dataset, X_train, y_train, k_folds, weights):
    
    try:
        scores = cross_validate(model.model, X_train, y_train, cv=k_folds, scoring=['accuracy', 'precision', 'recall', 'f1'])
        model.accuracy = scores['test_accuracy']
        model.precision = scores['test_precision']
        model.recall = scores['test_recall']
        model.f1 = scores['test_f1']
    except:
        k_fold = KFold(n_splits=k_folds, shuffle=True)
        accurancy = []
        precision = []
        recall = []
        f1 = []
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        for train, test in k_fold.split(X_train,y_train):
            nn = build_neural_network_2_layers(X_train.shape[1])
            nn.fit(X_train[train], y_train[train], epochs=400, class_weight=weights, batch_size=16)
            y_pred = nn.predict(X_train[test])
            y_pred = (y_pred > 0.5)
            # Calc metrics
            accurancy = np.append(accurancy, accuracy_score(y_train[test], y_pred))
            precision = np.append(precision, precision_score(y_train[test], y_pred))
            recall = np.append(recall, recall_score(y_train[test], y_pred))
            f1 = np.append(f1, f1_score(y_train[test], y_pred))
            model.model = nn
        model.accuracy = accurancy
        model.precision = precision
        model.recall = recall
        model.f1 = f1.mean()
    
    print("Model: "+model.model_name)
    print("Accuracy:", model.accuracy)
    print("Accuracy Mean:", model.accuracy.mean())
    print("Precision:", model.precision)
    print("Precision Mean:", model.precision.mean())
    print("Recall:", model.recall)
    print("Recall Mean:", model.recall.mean())
    print("F1:", model.f1)
    print("F1 Mean:", model.f1.mean())

    
    # result = permutation_importance(model.model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    # model_importances = pd.Series(result.importances_mean, index=dataset.drop(['Outcome'], axis=1).columns)

    # plt.clf()
    # fig, ax = plt.subplots()
    # model_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances of "+model.model_name)
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.savefig(save_dir+"/plots/"+'features_' + model.model_name + '.png')

def evaluate_models(models, dataset, X_test, y_test, weights):
    classification_report_txt = ""
    for model in models:
        # if model.skip:
        #     continue
        # model.model.evaluate(X_test, y_test)
        # Classification report
        y_pred = []
        if model.custom_evaluate_fn is not None:
            y_pred = model.custom_evaluate_fn(model.model, X_test)
        else:
            y_pred = model.model.predict(X_test)
        cr = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        # Set figure size
        fig, ax = plt.subplots(figsize=(10,10))
        # Set up the matplotlib figure
        sns.heatmap(cm,  cmap='Blues', annot=True, fmt='d',
                    xticklabels=['Yes', 'No'], yticklabels=['Yes', 'No'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(save_dir+"/plots/"+'confusion_matrix_' + model.model_name + '.png')
        # Roc curve
        plt.clf()
        # Calculate best threshold
       
        evaluate_models_with_k_folds(model,dataset,dataset.drop(['Outcome'], axis=1), dataset.Outcome, 5, weights)
        try:
            y_pred = model.model.predict_proba(X_test)[:,1]
        except:
            y_pred = model.model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # Calc best thresholds
        best_threshold_id = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_id]
        roc_auc = auc(fpr, tpr)
        plt.title('ROC curve for {}'.format(model.model_name))
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # Plot best_thresholds point
        plt.plot(fpr[best_threshold_id], tpr[best_threshold_id], 'gD', label='Best Threshold = %0.2f'% best_threshold)
        plt.savefig(save_dir+"/plots/"+'roc_' + model.model_name + '.png')

        classification_report_txt += model.model_name + '\n'
        # classification_report_txt += '5 folds: accuracy: ' + str(model.accuracy) +'; precision: '+str(model.precision)+'; recall: '+str(model.recall)+'; f1: '+str(model.f1)+'; best threshold: '+str(best_threshold)+ '\n\n'
        
    # Save classification report
    with open(save_dir+"/"+'classification_report.txt', 'w') as f:
        f.write(classification_report_txt)
    
dataset, X_train, X_test, y_train, y_test, weights = load_data(args.data_file_name,args.test_size)
models = build_classifier(X_train.shape[1])
train_models(models, X_train, y_train, weights)
evaluate_models(models,dataset, X_test, y_test, weights)
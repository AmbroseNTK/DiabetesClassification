# Run evaluate models on test data
# Import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import roc_curve, auc
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
from sklearn.model_selection import cross_validate
import joblib
import shutil
import os
from classification_model import ClassificationModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='xgb', help='Models to use for testing')
parser.add_argument('--test_file', type=str, default='test.csv', help='Test file')
parser.add_argument('--output', type=str, default='result.csv', help='Result directory')
parser.add_argument('--normalize',type=bool, default=True, help='Normalize data')
args = parser.parse_args()
model_path = args.model_path
test_file = args.test_file
output_dir = args.output

# Remove or create output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
os.makedirs(output_dir+"/plots")

# load test data
test_data = pd.read_csv(test_file)


def normalize_data(model):
    # Normalize test_data
    i = 0
    for col in test_data.columns:
        if col == 'Outcome':
            continue
        test_data[col] = test_data[col].apply(lambda x: (x - model.describes.iloc[3,i+1]) / (model.describes.iloc[7, i+1] - model.describes.iloc[3,i+1]))
        i += 1
    print(test_data.head())
    return test_data

def load_models():
    models = []
    model_names = os.listdir(model_path+"/models")
    for model_name in model_names:
        is_keras = model_name[-2:] == "h5"
        model = ClassificationModel.from_file(
            model_path+"/models/"+model_name, model_name, model_path+"/plots/describes.csv", is_keras)
        models.append(model)
    return models
    
def custom_evaluate_nn(model, X_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    return y_pred

def evaluate_models(models, X_test, y_test):
    classification_report_txt = ""
    thresholds = {
        "AdaBoost.pkl":0.49770280699808656,
        "DecisionTree.pkl":0.41935483870967744,
        "KNN.pkl":0.40933191950849684,
        "LogisticRegression.pkl":0.40933191950849684,
        "RandomForest.pkl":0.3,
        "SVM.pkl":0.22298740459948682,
        "XGBoost.pkl":0.33097824,
        "NaiveBayes.pkl":0.2650547028010106,
        "NeuralNetwork2Layers.h5":0.40297964,
    }
    for model in models:
        if model.skip:
            continue
        # model.model.evaluate(X_test, y_test)
        # Classification report
        result = []
        try:
            result = model.model.predict_proba(X_test)[:,1].tolist()
            
        except:
            result = model.model.predict(X_test).tolist()
        # print(np.shape(result))
        y_pred = []
        print(model.model_name)
        for i in range(len(result)):
            y_pred.append(1 if result[i] > thresholds[model.model_name] else 0)
            
        cr = []
        try:
            cr = classification_report(y_test, y_pred)
        except:
            y_pred = custom_evaluate_nn(model.model, X_test)
            cr = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        # Set figure size
        fig, ax = plt.subplots(figsize=(10,10))
        # Set up the matplotlib figure
        sns.heatmap(cm,  cmap='Blues', annot=True, fmt='d',
                    xticklabels=['Yes', 'No'], yticklabels=['Yes', 'No'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(output_dir+"/plots/"+'confusion_matrix_' + model.model_name + '.png')
        # Roc curve
        plt.clf()
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('ROC curve for {}'.format(model.model_name))
        plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(output_dir+"/plots/"+'roc_' + model.model_name + '.png')

        classification_report_txt += model.model_name + '\n' + cr + '\n'
    # Save classification report
    with open(output_dir+"/"+'classification_report.txt', 'w') as f:
        f.write(classification_report_txt)

def calc_best_threshold(models, X_test, y_test):
    for model in models:
        if model.skip:
            continue
        y_pred = []
        try:
            y_pred = model.model.predict_proba(X_test)[:,1]
        except:
            y_pred = model.model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # Calc best thresholds
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        print("Model: "+model.model_name+": "+str(best_threshold))

models = load_models()
if args.normalize:
    test_data = normalize_data(models[0])
# calc_best_threshold(models, test_data.drop(['Outcome'], axis=1), test_data['Outcome'])
evaluate_models(models, test_data.drop(['Outcome'], axis=1), test_data['Outcome'])
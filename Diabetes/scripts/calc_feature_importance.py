# Calculate feature importance for a given model with permutation importance
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
from sklearn.inspection import plot_partial_dependence, permutation_importance
from sklearn.metrics import accuracy_score

model_dir = "./model_15k/models/NeuralNetwork2Layers.h5"

# load model
model = tf.keras.models.load_model(model_dir)

# load X_train and y_train
X_test = pd.read_csv('../dataset/PIMA_15k_selected_FG_norm.csv')
y_test = X_test['Outcome']
X_test = X_test.drop(columns=['Outcome'])

# Shuffle a column's data
def shuffle_column(df, column):
    df[column] = np.random.permutation(df[column])
    return df

y_pred_origin = model.predict(X_test)
y_pred_origin = [1 if x >= 0.5 else 0 for x in y_pred_origin]
score_origin = accuracy_score(y_test, y_pred_origin)
print(score_origin)
score_list = []
for k in range(10):
    scores = []
    for i in range(0, len(X_test.columns)):
        X_test_shuffled = shuffle_column(X_test.copy(), X_test.copy().columns[i])
        y_pred = model.predict(X_test_shuffled)
        y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
        score = accuracy_score(y_test, y_pred)
        scores.append(np.power(score - score_origin,2))
    score_list.append(scores)

# Calculate mean
score_list = np.array(score_list)
score_mean = np.mean(score_list, axis=0)

print(score_list)
print(score_mean)
# plot bar chart
plt.bar(X_test.columns, score_mean)
plt.show()

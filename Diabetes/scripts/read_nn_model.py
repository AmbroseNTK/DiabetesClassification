# Visualize the weights of a neural network
# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
import joblib

# # model dir
# model_dir = './model_15k/models/LogisticRegression.pkl'

# model = joblib.load(model_dir)
# print(model.coef_)
# print(model.intercept_)

# # load model
# model = tf.keras.models.load_model(model_dir)


# # visual the weights

# # get weights
# weights = model.get_weights()
# # Save to dataframe
# df = pd.DataFrame(weights)
# # Save to csv
# df.to_csv('weights.csv')


    
class VisualizeTrainingSteps(keras.callbacks.Callback):

    def __init__(self, model, dir, name, steps):
        self.model = model
        self.dir = dir
        self.name = name
        self.steps = steps

    def visualize_weights(self, model,dir, name, step):
        weights = model.get_weights()
        print(weights)
        # Plot all layers in one plot
        fig, axs = plt.subplots(len(weights), 1, figsize=(10, 10))
        # save to csv
        for i, ax in enumerate(axs):
            # Plot the weights
            w = weights[i]
            if w.ndim == 1:
                w = np.expand_dims(weights[i],axis=0)
            
            ax.imshow(w, cmap='afmhot')
            # Plot the weight names
            # ax.set_title(model.layers[i].name)
            # add axis labels
            if i % 2 == 0:
                ax.set_xlabel('Neurons')
                ax.set_ylabel('Features')
                # save to csv
                np.savetxt(dir + '/' + name + '_' + str(step) + '_'  + 'weights.csv', w, delimiter=',')
            else:
                ax.set_xlabel('Neuron thresholds')
                # save to csv
                np.savetxt(dir + '/' + name + '_' + str(step) + '_' + str(i) + 'thresholds.csv', w, delimiter=',')
            # add axis values
            ax.set_xticks(np.arange(w.shape[1]))
            ax.set_yticks(np.arange(w.shape[0]))


        plt.tight_layout()
        # save the figure
        plt.savefig(dir+'/'+name+'_'+str(step)+'.png')

    # def on_train_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))
        self.visualize_weights(self.model, self.dir, self.name, -1)

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
        if epoch == 0 or epoch % self.steps == 0:
            self.visualize_weights(self.model, self.dir, self.name, epoch)
# visualize_weights(model, 'NN2L',0)

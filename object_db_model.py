import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from keras.models import Sequential

from keras.optimizers import Adam,SGD

from keras.callbacks import ModelCheckpoint,TensorBoard

from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from utils import INPUT_SHAPE, batch_generator

import argparse

import os
from time import time

#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)
 
def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(args.data_dir, 'Drive_track_db2.csv'))
    #get the 1st column of the table containing images as input
    X = data_df[data_df.columns[0]].values
    #get the next 2 columns as the outputs
    y = data_df[[data_df.columns[1],data_df.columns[2]]].values
    # split the data into training and testing data.
    # training data will be used for training and later testing/validation data, 
    # unseen by the model is used for testing.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    #Sequential - layers are joined one after another
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0,input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', strides=(1, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', strides=(1, 1)))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.4))
    model.add(Flatten())
    #model.add(Dense(1280, use_bias=True, activation='elu'))
    model.add(Dense(600, use_bias=True,activation='relu'))
    model.add(Dense(300, use_bias=True,activation='relu'))
    #model.add(Dense(80, use_bias=True,activation='tanh'))
    model.add(Dense(2,activation='tanh'))
    model.summary()

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #checkpoints stores the model and its parameters according to the best training accuracy
    """checkpoint = ModelCheckpoint('log1/model-{epoch:03d}.hdf5',
                                                         monitor='val_loss',
                                                         verbose=0,
                                                         save_best_only=args.save_best_only,
                                                         mode='auto')"""
    #tensorBoard checkpoints helps to visualize the m=network model, parameters, graph etc.
    checkpoint =TensorBoard(log_dir="log/model_train{}".format(time()),histogram_freq=0,
                                                         write_graph=True,write_grads = True, write_images=True)
    #Brings the model togather, sets the loss function and the metrics .
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=args.learning_rate, momentum=0.9),
                                                            metrics =['accuracy'])
    #Fit/fit_generator controls the whole operation of fitting the dataset to the output
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        steps_per_epoch = args.steps_per_epoch,
                        epochs = args.nb_epoch,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        validation_steps=3 ,#len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1,
                        max_queue_size=1)
    #save the weights and the model
    model.save_weights('object_db_weights_tracks2.h5',overwrite=True)
    model.save('object_db_model_tracks2.h5',overwrite=True)
    print("Model = object_db_model_tracks2.h5")
#for command line args



def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Object Database Training Program')
    #give data directory 
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='C:/Users/ankur/Desktop/Ankur/Python_Project')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.1)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='steps_per_epoch',       dest='steps_per_epoch',   type=int,   default=250)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=0.005)
    parser.add_argument('-z', help='optimizer',             dest='optimizer_',        type=str,   default='SGD')
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)
    #load data
    data = load_data(args)
    #print("Data = {}".format(data))
    #build model
    model = build_model(args)
    #model.load_weights('object_db_weights_tracks1.h5')
    #train model on data, it saves as model.h5 
    train_model(model, args, *data)


if __name__ == '__main__':
    main()


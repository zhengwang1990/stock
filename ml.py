import argparse
import numpy as np
import pandas as pd
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import train_test_split
from tabulate import tabulate

DEFAULT_DATA_FILE = 'simulate_stats.csv'
NON_ML_FEATURE_COLUMNS = ['Gain', 'Symbol', 'Date']
TRAIN_ITER = 2

class ML(object):

    def __init__(self, model=None, data_files=None):
        data_files = data_files or [DEFAULT_DATA_FILE]
        self.model = model
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.df = pd.concat([pd.read_csv(
            os.path.join(self.root_dir, utils.DATA_DIR, data_file))
            for data_file in data_files])
        self.ml_features = [col for col, _ in self.df.iteritems()
                            if col not in NON_ML_FEATURE_COLUMNS]
        self.X, self.y = [], []
        for row in self.df.itertuples():
            x_value = [getattr(row, col) for col in self.ml_features]
            y_value = row.Gain / 0.05 if np.abs(row.Gain) < 0.05 else np.sign(row.Gain)
            self.X.append(x_value)
            self.y.append(y_value)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=0)

    def create_model(self):
        x_dim = len(self.df.columns) - 3
        model = keras.Sequential([
            keras.layers.Dense(100, activation='relu',
                               input_shape=(x_dim,)),
            keras.layers.Dense(1, activation='tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def fit_model(self, model):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50, restore_best_weights=True)
        model.fit(self.X_train, self.y_train, batch_size=512, epochs=1000,
                  validation_data=(self.X_test, self.y_test),
                  callbacks=[early_stopping])

    def evaluate(self, model, plot=False):
        y_pred = model.predict(self.X)
        boundary_90 = np.percentile(y_pred, 90)
        precision_90 = get_precision(self.y, y_pred, boundary_90)
        precision_model = get_precision(self.y, y_pred, 0)
        output = [['Precision_90', precision_90],
                  ['Model Precision', precision_model],
                  ['Boundary_90', boundary_90]]
        print(tabulate(output, tablefmt='grid'))
        if plot:
            plt.figure()
            plt.plot(y_pred, self.y, 'o', markersize=3)
            plt.xlabel('Predicted')
            plt.ylabel('Truth')
            plt.plot([np.min(y_pred), np.max(y_pred)], [0, 0], '--')
            plt.plot([0, 0], [np.min(self.y), np.max(self.y)], '--', label='0')
            plt.plot([boundary_90, boundary_90], [np.min(self.y), np.max(self.y)], '--', label='Percentile 90')
            plt.legend()
            plt.show()
        return precision_90

    def train(self):
        precision_max, model_max = 0, None
        for _ in range(TRAIN_ITER):
            model = self.create_model()
            self.fit_model(model)
            precision = self.evaluate(model)
            if precision > precision_max:
                precision_max = precision
                model_max = model
        model_name = 'model_p%d.hdf5' % (int(precision_max * 1E6),)
        model_max.save(os.path.join(self.root_dir, utils.MODELS_DIR, model_name))
        print(utils.get_header('Final Model'))
        self.evaluate(model_max, plot=True)

    def load(self):
        model = keras.models.load_model(
            os.path.join(self.root_dir, utils.MODELS_DIR, self.model))
        model.summary()
        self.evaluate(model, plot=True)


def get_precision(y_true, y_pred, boundary):
    tp, tn, fp, fn = 0, 0, 0, 0
    for pi, yi in zip(y_pred, y_true):
        if pi >= boundary:
            if yi >= 0:
                tp += 1
            else:
                fp += 1
    precision = tp / (tp + fp + 1E-7)
    return precision


def main():
    parser = argparse.ArgumentParser(description='Stock trading ML model.')
    parser.add_argument('--model', default=None,
                        help='Model name to load')
    parser.add_argument('--data_files', default=[], nargs='*',
                        help='Data to train on.')
    args = parser.parse_args()
    ml = ML(args.model, args.data_files)
    print(args.data_files)
    if args.model:
        ml.load()
    else:
        ml.train()


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import os
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import train_test_split
from tabulate import tabulate

DATA_FILE = 'simulate_stats.csv'


def read_df():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, utils.OUTPUTS_DIR, DATA_FILE))
    return df


def plot_data():
    df = read_df()
    y = df.get('Gain')
    for key, _ in df.iteritems():
        if key not in ('Gain', 'Symbol', 'Date'):
            x = df.get(key)
            plt.figure()
            plt.plot(x, y, 'o', markersize=3)
            plt.plot([np.min(x), np.max(x)], [0, 0], '--')
            plt.title(key + ' v.s. Gain')
            plt.show()


def load_data():
    df = read_df()
    keys = [key for key, _ in df.iteritems() if key not in ('Gain', 'Symbol', 'Date')]
    x, y = [], []
    for row in df.itertuples():
        x_value = [getattr(row, key) for key in keys]
        y_value = row.Gain / 5 if np.abs(row.Gain) < 5 else np.sign(row.Gain)
        x.append(x_value)
        y.append(y_value)
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    return x_train, x_test, y_train, y_test


def precision_favored_loss(y_true, y_pred):
    fp = (1 + y_pred) * (1 - y_true)
    fn = (1 - y_pred) * (1 + y_true)
    loss = K.mean(K.pow(fp, 2) + K.pow(fn, 2))
    return loss


def get_model():
    df = read_df()
    x_dim = len(df.columns) - 3
    model = keras.Sequential([
        keras.layers.Dense(100, activation='relu',
                           input_shape=(x_dim,)),
        keras.layers.Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss=precision_favored_loss)
    model.summary()
    return model


def train_model(x_train, x_test, y_train, y_test, model):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=25, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=256, epochs=500,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.save(os.path.join(dir_path, utils.MODELS_DIR, 'model.hdf5'))


def load_model(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model = keras.models.load_model(
        os.path.join(dir_path, utils.MODELS_DIR, name),
        custom_objects={'precision_favored_loss': precision_favored_loss})
    return model


def get_measures(p, y, boundary):
    tp, tn, fp, fn = 0, 0, 0, 0
    for pi, yi in zip(p, y):
        if pi >= boundary:
            if yi >= 0:
                tp += 1
            else:
                fp += 1
        else:
            if yi > 0:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp + 1E-7)
    recall = tp / (tp + fn + 1E-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # print('Boundary: %.3f, Precision: %.5f, Recall: %.2e' % (boundary, precision, recall))
    return precision, recall, accuracy


def predict(x, y, model, plot=False):
    boundary = 0
    p = model.predict(x)
    boundary_70 = np.percentile(p, 70)
    boundary_90 = np.percentile(p, 90)
    boundary_95 = np.percentile(p, 95)
    precision_70, recall_70, accuracy_70 = get_measures(p, y, boundary_70)
    precision_90, recall_90, accuracy_90 = get_measures(p, y, boundary_90)
    precision_95, recall_95, accuracy_95 = get_measures(p, y, boundary_95)
    precision_buy_all = len(y[y > 0]) / len(y)
    precision_model, recall_model, accuracy_model = get_measures(p, y, 0)
    output = [['Precision_70:', precision_70],
              ['Precision_90:', precision_90],
              ['Precision_95:', precision_95],
              ['Buy All Precision:', precision_buy_all],
              ['Model Precision:', precision_model],
              ['Boundary_70:', boundary_70],
              ['Boundary_90:', boundary_90]]
    print(tabulate(output, tablefmt='grid'))

    if plot:
        plt.figure()
        plt.plot(p, y, 'o', markersize=3)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.plot([np.min(p), np.max(p)], [0, 0], '--')
        plt.plot([0, 0], [np.min(y), np.max(y)], '--', label='0')
        plt.plot([boundary_70, boundary_70], [np.min(y), np.max(y)], '--', label='Percentile 70')
        plt.plot([boundary_90, boundary_90], [np.min(y), np.max(y)], '--', label='Percentile 90')
        plt.plot([boundary_95, boundary_95], [np.min(y), np.max(y)], '--', label='Percentile 95')
        plt.legend()
        plt.show()
    return precision_90


def train_once():
    x_train, x_test, y_train, y_test = load_data()
    # model = get_model()
    # train_model(x_train, x_test, y_train, y_test, model)
    model = load_model('model_p612804.hdf5')
    print(utils.get_header('Training Split'))
    predict(x_train, y_train, model)
    print(utils.get_header('Testing Split'))
    predict(x_test, y_test, model, plot=True)


def train_loop():
    x_train, x_test, y_train, y_test = load_data()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    precision_max = 0
    for _ in range(10):
        model = get_model()
        train_model(x_train, x_test, y_train, y_test, model)
        precision = predict(x_test, y_test, model)
        if precision > precision_max:
            precision_max = precision
            os.rename(os.path.join(dir_path, utils.MODELS_DIR, 'model.hdf5'),
                      os.path.join(dir_path, utils.MODELS_DIR, 'model_p%d.hdf5' % (int(precision * 1000000),)))


def main():
    # train_loop()
    train_once()


if __name__ == '__main__':
    main()

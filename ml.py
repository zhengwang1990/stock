import numpy as np
import pandas as pd
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from common import *
from sklearn.model_selection import train_test_split
from tabulate import tabulate

DATA_FILE = 'simulate_stats0.csv'

MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')

def read_df():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, OUTPUTS_DIR, DATA_FILE))
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
    fp = K.mean((1 + y_pred) * (1 - y_true))
    fn = K.mean((1 - y_pred) * (1 + y_true))
    loss = K.pow(fp, 3) + K.pow(fn, 2)
    return loss


def get_model():
    df = read_df()
    x_dim = len(df.columns) - 3
    model = keras.Sequential([
        keras.layers.Input(shape=(x_dim,)),
        keras.layers.Dense(40, activation='relu',
                           input_shape=(x_dim,)),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(40, activation='relu'),
        keras.layers.Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss=precision_favored_loss)
    model.summary()
    return model


def train_model(x, y, model):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True)
    model.fit(x, y, epochs=100, callbacks=[early_stopping])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.save(os.path.join(dir_path, MODELS_DIR, 'model.hdf5'))


def load_model(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model = keras.models.load_model(
        os.path.join(dir_path, MODELS_DIR, name),
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
    #print('Boundary: %.3f, Precision: %.5f, Recall: %.2e' % (boundary, precision, recall))
    return precision, recall, accuracy


def predict(x, y, model):
    boundary = 0
    p = model.predict(x)
    #chosen_precision, chosen_recall, chosen_accuracy, chosen_boundary = 0, 0, 0, 0
    #for boundary in tqdm(np.arange(np.min(p), np.max(p), 0.001), ncols=80):
    boundary_50 = np.percentile(p, 50)
    boundary_90 = np.percentile(p, 90)
    boundary_95 = np.percentile(p, 95)
    precision_50, recall_50, accuracy_50 = get_measures(p, y, boundary_50)
    precision_90, recall_90, accuracy_90 = get_measures(p, y, boundary_90)
    precision_95, recall_95, accuracy_95 = get_measures(p, y, boundary_95)
    precision_buy_all = len(y[y > 0]) / len(y)
    precision_model, recall_model, accuracy_model = get_measures(p, y, 0)
    output = [['Precision_50:', precision_50],
              ['Precision_90:', precision_90],
              ['Precision_95:', precision_95],
              ['Buy All Precision:', precision_buy_all],
              ['Model Precision:', precision_model]]
    print(tabulate(output, tablefmt='grid'))

    ind = np.random.choice(len(p), 500)
    plt.figure()
    plt.plot(p[ind], y[ind], 'o', markersize=3)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.plot([np.min(p), np.max(p)], [0, 0], '--')
    plt.plot([0, 0], [np.min(y), np.max(y)], '--')
    plt.plot([boundary_50, boundary_50], [np.min(y), np.max(y)], '--')
    plt.plot([boundary_90, boundary_90], [np.min(y), np.max(y)], '--')
    plt.plot([boundary_95, boundary_95], [np.min(y), np.max(y)], '--')
    plt.show()


def main():
    x_train, x_test, y_train, y_test = load_data()
    model = get_model()
    train_model(x_train, y_train, model)
    #model = load_model('model_precision_630958.hdf5')
    predict(x_test, y_test, model)


if __name__ == '__main__':
    main()

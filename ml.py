import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import utils
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
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
    scalars = {}
    for key in keys:
        scalars[key] = 1.0 / np.std(df.get(key))
    for row in df.itertuples():
        x_value = [getattr(row, key) * scalars[key] for key in keys]
        y_value = row.Gain * 100
        x.append(x_value)
        y.append(y_value)
    x = np.array(x)
    y = np.array(y)
    return x, y, keys, scalars


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
    return precision, recall, accuracy


def train():
    X, y, keys, scalars = load_data()
    reg = Lasso(alpha=0.01, max_iter=10000)
    #reg = LinearRegression()
    reg.fit(X, y)
    p = reg.predict(X)

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

    coefficients = {}
    for key, coefficient in zip(keys, reg.coef_):
        coefficients[key] = coefficient
    print('REGRESSION_COEFFICIENT = {')
    for k, v in coefficients.items():
        print("    '%s': %e," % (k, v * scalars[k]))
    print('}')
    print('REGRESSION_INTERCEPT =', reg.intercept_)

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


def main():
    train()


if __name__ == '__main__':
    main()

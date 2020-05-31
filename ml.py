import argparse
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import train_test_split
from tabulate import tabulate

NON_ML_FEATURE_COLUMNS = ['Gain', 'Symbol', 'Date']
DEFAULT_TRAIN_ITER = 1


class ML(object):

    def __init__(self, data_files, model=None, train_iter=DEFAULT_TRAIN_ITER):
        data_files = data_files
        self.model = model
        self.train_iter = train_iter
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.df = pd.concat([pd.read_csv(data_file) for data_file in data_files])
        self.X, self.T, self.y = [], [], []
        for _, row in self.df.iterrows():
            x_value = [row[col] for col in utils.ML_TECH_FEATURES]
            t_value = [[row[col]] for col in utils.ML_TIME_FEATURES]
            gain = row['Gain']
            y_value = gain / 0.05 if np.abs(gain) < 0.05 else np.sign(gain)
            self.X.append(x_value)
            self.T.append(t_value)
            self.y.append(y_value)
        self.X = np.array(self.X)
        self.T = np.array(self.T)
        self.y = np.array(self.y, dtype=np.float32)
        self.w = 0.3 + np.arange(len(self.df)) / len(self.df) * 0.7
        split_result = train_test_split(self.X, self.T, self.y, self.w, test_size=0.1, random_state=0)
        self.X_train, self.X_test = split_result[0:2]
        self.T_train, self.T_test = split_result[2:4]
        self.y_train, self.y_test = split_result[4:6]
        self.w_train, self.w_test = split_result[6:8]

    @staticmethod
    def loss_function(c_layer):
        def _loss(y_true, y_pred):
            if not tf.is_tensor(y_pred):
                y_pred = tf.constant(y_pred)
            return K.mean(c_layer * K.square(y_pred - y_true) + 0.15 * (1 - c_layer), axis=-1)

        return _loss

    @staticmethod
    def create_model():
        x_input = keras.layers.Input(shape=(len(utils.ML_TECH_FEATURES, )), name='x_input')
        x = keras.layers.Dense(50, activation='relu', name='x_dense')(x_input)
        x = keras.layers.Dropout(0.2, name='x_dropout')(x)

        t_input = keras.layers.Input(shape=(len(utils.ML_TIME_FEATURES), 1), name='t_input')
        t = keras.layers.Conv1D(5, kernel_size=3, activation='relu', use_bias=False, name='t_conv_1')(t_input)
        t = keras.layers.MaxPool1D(pool_size=2, name='t_pool_1')(t)
        t = keras.layers.Conv1D(8, kernel_size=3, activation='relu', use_bias=False, name='t_conv_2')(t)
        t = keras.layers.MaxPool1D(pool_size=2, name='t_pool_2')(t)
        t = keras.layers.Flatten(name='t_flatten')(t)
        t = keras.layers.Dropout(0.5, name='t_dropout')(t)

        info = keras.layers.concatenate([x, t])

        c = keras.layers.Dense(1, activation='sigmoid',)(info)
        r = keras.layers.Dense(1, activation='tanh', name='regression')(info)

        main_model = keras.Model(inputs=[x_input, t_input], outputs=r)
        save_model = keras.Model(inputs=[x_input, t_input], outputs=[r, c])
        main_model.compile(optimizer='adam', loss=ML.loss_function(c),
                           experimental_run_tf_function=False)
        main_model.summary()
        return main_model, save_model

    def fit_model(self, model):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50, restore_best_weights=True)
        model.fit([self.X_train, self.T_train], self.y_train, batch_size=512, epochs=5000,
                  sample_weight=self.w_train,
                  validation_data=([self.X_test, self.T_test], self.y_test, self.w_test),
                  callbacks=[early_stopping])

    def evaluate(self, model):
        y_pred, c_pred = model.predict([self.X, self.T])
        y_true = self.y
        y_boundary, c_boundary = 0, 0.5
        precision, recall = get_accuracy(y_true, y_pred, c_pred, y_boundary, c_boundary)
        baseline = np.sum(y_true > 0) / (np.sum(y_true > 0) + np.sum(y_true < 0))
        print(utils.get_header('Examples'))
        example_count = 30
        examples = []
        for i in range(example_count):
            correct = 'N/A'
            if c_pred[i] >= c_boundary:
                if ((y_true[i] > y_boundary and y_pred[i] > y_boundary) or
                        (y_true[i] < -y_boundary and y_pred[i] < -y_boundary)):
                    correct = 'Y'
                elif y_true[i] == 0:
                    correct = 'I'
                else:
                    correct = 'N'

            examples.append([y_true[i], y_pred[i], c_pred[i], correct])
        print(tabulate(examples, tablefmt='grid', headers=['Truth', 'Prediction', 'Confidence', 'Correct']))
        print(utils.get_header('Model Stats'))
        output = [['Model Precision', '%.2f%%' % (precision * 100,)],
                  ['Model Recall', '%.2f%%' % (recall * 100,)],
                  ['Baseline Precision', '%.2f%%' % (baseline * 100,)],
                  ['Positive Count', np.sum(np.logical_and(y_pred > y_boundary, c_pred > c_boundary))],
                  ['Classification Filter Rate', '%.2f%%' % (np.sum(c_pred <= c_boundary) / len(c_pred) * 100,)]]
        print(tabulate(output, tablefmt='grid'))
        plot(y_true, y_pred, c_pred, c_boundary)
        return precision

    def train(self):
        model, save_model = self.create_model()
        self.fit_model(model)
        precision = self.evaluate(save_model)
        model_name = 'model_p%d.hdf5' % (int(precision * 1E6),)
        save_model.save(os.path.join(self.root_dir, utils.MODELS_DIR, model_name))

    def load(self):
        model = keras.models.load_model(
            os.path.join(self.root_dir, utils.MODELS_DIR, self.model))
        model.summary()
        self.evaluate(model)


def plot(y_true, y_pred, c_pred, c_boundary):
    points = {}
    y_min = np.percentile(y_pred, 10)
    y_max = np.percentile(y_pred, 90)
    granularity = (y_max - y_min) / 10
    for pi, ci, yi in zip(y_pred, c_pred, y_true):
        if ci < c_boundary:
            continue
        p = (int(pi / granularity), int(yi / 0.1))
        points[p] = points.get(p, 0) + 1
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17, 17))
    max_size = max(points.values())
    for point, size in points.items():
        ax1.plot([point[0] * granularity], [point[1] * 0.1], 'o', markersize=size / max_size * 12, c='C0')
    ax1.grid('--')
    ax1.set_xlim((y_min-0.1, y_max+0.1))
    ax1.set_ylim((-1.5, 1.5))
    ax1.set_xlabel('Prediction')
    ax1.set_ylabel('Truth')
    ax1.set_title('Scatter Plot')

    ax2.hist(c_pred, bins=20)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')

    ax3.hist(y_pred, bins=20)
    ax3.set_xlabel('Prediction')
    ax3.set_ylabel('Count')
    ax3.set_title('Prediction Distribution')

    ax4.hist(y_true, bins=20)
    ax4.set_xlabel('Label')
    ax4.set_ylabel('Count')
    ax4.set_title('Truth Distribution')
    plt.show()


def get_accuracy(y_true, y_pred, c_pred, y_boundary, c_boundary):
    tp, tn, fp, fn = 0, 0, 0, 0
    for pi, ci, yi in zip(y_pred, c_pred, y_true):
        if ci < c_boundary:
            continue
        if pi > y_boundary:
            if yi > y_boundary:
                tp += 1
            elif yi < -y_boundary:
                fp += 1
        else:
            if yi > y_boundary:
                fn += 1
    precision = tp / (tp + fp + 1E-7)
    recall = tp / (tp + fn + 1E-7)
    return precision, recall


def main():
    parser = argparse.ArgumentParser(description='Stock trading ML model.')
    parser.add_argument('--model', default=None,
                        help='Model name to load')
    parser.add_argument('--data_files', required=True, nargs='*',
                        help='Data to train on.')
    parser.add_argument('--train_iter', type=int, default=DEFAULT_TRAIN_ITER,
                        help='Iterations in training.')
    args = parser.parse_args()
    ml = ML(args.data_files, args.model, args.train_iter)
    print(args.data_files)
    if args.model:
        ml.load()
    else:
        ml.train()


if __name__ == '__main__':
    main()

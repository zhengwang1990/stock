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
            if gain >= 0.01:
                y_value = [1, 0, 0]
            elif gain <= -0.01:
                y_value = [0, 1, 0]
            else:
                y_value = [0, 0, 1]
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
            return K.mean(c_layer * K.square(y_pred - y_true) + 0.8 * (1 - c_layer), axis=-1)

        return _loss

    @staticmethod
    def create_model():
        x_input = keras.layers.Input(shape=(len(utils.ML_TECH_FEATURES, )), name='x_input')
        x = keras.layers.Dense(50, activation='relu', name='x_dense_1')(x_input)
        x = keras.layers.Dense(20, activation='relu', name='x_dense_2')(x)
        x = keras.layers.Dense(10, activation='relu', name='x_dense_3')(x)
        x = keras.layers.Dropout(0.2, name='x_dropout')(x)

        t_input = keras.layers.Input(shape=(len(utils.ML_TIME_FEATURES), 1), name='t_input')
        t = keras.layers.Conv1D(4, kernel_size=3, activation='relu', use_bias=False, name='t_conv_1')(t_input)
        t = keras.layers.Conv1D(8, kernel_size=3, activation='relu', use_bias=False, name='t_conv_2')(t)
        t = keras.layers.MaxPool1D(pool_size=2, name='t_pool_1')(t)
        t = keras.layers.Conv1D(8, kernel_size=3, activation='relu', use_bias=False, name='t_conv_3')(t)
        t = keras.layers.Conv1D(16, kernel_size=3, activation='relu', use_bias=False, name='t_conv_4')(t)
        t = keras.layers.MaxPool1D(pool_size=2, name='t_pool_2')(t)
        t = keras.layers.Conv1D(32, kernel_size=3, activation='relu', use_bias=False, name='t_conv_5')(t)
        t = keras.layers.Conv1D(64, kernel_size=3, activation='relu', use_bias=False, name='t_conv_6')(t)
        t = keras.layers.MaxPool1D(pool_size=2, name='t_pool_3')(t)
        t = keras.layers.Flatten(name='t_flatten')(t)
        t = keras.layers.Dropout(0.3, name='t_dropout')(t)

        info = keras.layers.concatenate([x, t])

        r = keras.layers.Dense(3, activation='softmax', name='classification',
                               kernel_regularizer=keras.regularizers.l2(0.1))(info)

        model = keras.Model(inputs=[x_input, t_input], outputs=r)

        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def fit_model(self, model):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit([self.X_train, self.T_train], self.y_train, batch_size=512, epochs=1000,
                  sample_weight=self.w_train,
                  validation_data=([self.X_test, self.T_test], self.y_test, self.w_test),
                  callbacks=[early_stopping])

    def evaluate(self, model):
        y_pred = model.predict([self.X, self.T])
        y_true = self.y
        precision, recall, accuracy = get_accuracy(y_true, y_pred)

        c_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for yi, pi in zip(y_true, y_pred):
            c_true = np.argmax(yi)
            c_pred = np.argmax(pi)
            c_matrix[c_true][c_pred] += 1
        pos_true = np.sum(c_matrix[0])
        neg_true = np.sum(c_matrix[2])
        pos_pred = c_matrix[0][0] + c_matrix[1][0] + c_matrix[2][0]
        baseline = pos_true / (pos_true + neg_true + 1E-7)
        print(utils.get_header('Examples'))
        example_count = 30
        examples = []
        for i in range(example_count):
            if np.argmax(y_true[i]) == np.argmax(y_pred[i]):
                correct = 'Y'
            elif np.argmax(y_true[i]) == 1:
                correct = 'I'
            else:
                correct = 'N'
            examples.append([y_true[i], y_pred[i], correct])
        print(tabulate(examples, tablefmt='grid', headers=['Truth', 'Prediction', 'Correct']))
        print(utils.get_header('Classification Matrix'))
        matrix = [['', 'Prediction Gain', 'Prediction Flat', 'Prediction Loss'],
                  ['Truth Gain'] + c_matrix[0],
                  ['Truth Flat'] + c_matrix[1],
                  ['Truth Loss'] + c_matrix[2]]
        print(tabulate(matrix, tablefmt='grid'))
        print(utils.get_header('Model Stats'))
        output = [['Precision', '%.2f%%' % (precision * 100,)],
                  ['Recall', '%.2f%%' % (recall * 100,)],
                  ['Accuracy', '%.2f%%' % (accuracy * 100,)],
                  ['Baseline Precision', '%.2f%%' % (baseline * 100,)],
                  ['Positive Count', pos_pred]]
        print(tabulate(output, tablefmt='grid'))
        #plot(y_true, y_pred)
        return precision

    def train(self):
        model = self.create_model()
        self.fit_model(model)
        precision = self.evaluate(model)
        model_name = 'model_p%d.hdf5' % (int(precision * 1E6),)
        model.save(os.path.join(self.root_dir, utils.MODELS_DIR, model_name))

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


def get_accuracy(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for pi, yi in zip(y_pred, y_true):
        yc = np.argmax(yi)
        pc = np.argmax(pi)
        if pc == 0:
            if yc == 0:
                tp += 1
            elif yc == 2:
                fp += 1
        elif pc == 2:
            if yc == 0:
                fn += 1
            elif yc == 2:
                tn += 1
    precision = tp / (tp + fp + 1E-7)
    recall = tp / (tp + fn + 1E-7)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1E-7)
    return precision, recall, accuracy


def main():
    parser = argparse.ArgumentParser(description='Stock trading ML model.')
    parser.add_argument('--model', default=None,
                        help='Model name to load')
    parser.add_argument('--data_files', required=True, nargs='+',
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

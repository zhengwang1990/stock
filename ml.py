import argparse
import numpy as np
import logging
import os
import pandas as pd
import pickle
import utils
from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn import metrics
from tabulate import tabulate

NON_ML_FEATURE_COLUMNS = ['Gain', 'Symbol', 'Date']


def print_metrics(y_true, y_pred, y_meta, title_prefix=''):
    outputs = []
    confusion_matrix_main = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    confusion_table_main = [['', 'Predict Short', 'Predict Long'],
                            ['True Short', confusion_matrix_main[0][0], confusion_matrix_main[0][1]],
                            ['True Long', confusion_matrix_main[1][0], confusion_matrix_main[1][1]]]
    outputs.extend([utils.get_header(title_prefix + 'Main Confusion Matrix'),
                    tabulate(confusion_table_main, tablefmt='grid')])

    threshold = np.percentile(y_meta, 90)
    indices = [i for i in range(len(y_meta)) if y_meta[i] > threshold]
    dropped = len(y_meta) - len(indices)
    outputs.append('Threshold: %.2f. %d (%.1f%%) samples dropped. %d (%.1f%%) samples preserved.' % (
        threshold, dropped, dropped / len(y_meta) * 100, len(indices), len(indices) / len(y_meta) * 100))
    y_true_meta = y_true[indices]
    y_pred_meta = y_pred[indices]
    confusion_matrix_final = metrics.confusion_matrix(y_true_meta, y_pred_meta, labels=[0, 1])
    confusion_table_final = [['', 'Predict Short', 'Predict Long'],
                             ['True Short', confusion_matrix_final[0][0], confusion_matrix_final[0][1]],
                             ['True Long', confusion_matrix_final[1][0], confusion_matrix_final[1][1]]]
    outputs.extend([utils.get_header(title_prefix + 'Final Confusion Matrix'),
                    tabulate(confusion_table_final, tablefmt='grid')])

    accuracy_main = metrics.accuracy_score(y_true, y_pred)
    accuracy_final = metrics.accuracy_score(y_true_meta, y_pred_meta)
    benchmark = np.sum(y_true == 1) / len(y_true)
    benchmark = benchmark if benchmark > 0.5 else 1 - benchmark
    accuracy_table = [['Accuracy Main', accuracy_main],
                      ['Accuracy Final', accuracy_final],
                      ['Benchmark', benchmark]]
    outputs.extend([utils.get_header(title_prefix + 'Accuracy'),
                    tabulate(accuracy_table, tablefmt='grid')])

    logging.info('\n'.join(outputs))
    return accuracy_final


class ML(object):

    def __init__(self, data_files, model_suffix=None):
        data_files = data_files
        self.model_suffix = model_suffix
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        logging.info('Reading csv data...')
        self.df = pd.concat([pd.read_csv(data_file) for data_file in data_files])
        self.X, self.y = [], []
        self.symbols = []
        logging.info('Processing data...')
        for _, row in self.df.iterrows():
            gain = row['Gain']
            if gain > 0:
                y_value = 1
            elif gain < 0:
                y_value = 0
            else:
                continue
            x_value = [row[col] for col in utils.ML_FEATURES]
            self.X.append(x_value)
            self.y.append(y_value)
            self.symbols.append([row['Symbol'], row['Date'], gain])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.w = 0.3 + np.arange(len(self.X)) / len(self.X) * 0.7
        logging.info('%d data samples loaded', len(self.X))
        self.hyper_parameters = {'max_depth': len(utils.ML_FEATURES),
                                 'min_samples_leaf': 0.005,
                                 'n_jobs': -1}
        logging.info('Model hyper-parameters: %s', self.hyper_parameters)

    def k_fold_cross_validation(self, k):
        main_model = ensemble.RandomForestClassifier(**self.hyper_parameters)
        meta_model = ensemble.RandomForestRegressor(**self.hyper_parameters)
        k_fold = KFold(n_splits=k, shuffle=True, random_state=0)
        fold = 1
        accuracy_sum = 0
        accuracy_table = []
        for train_index, test_index in k_fold.split(self.X):
            X_train = self.X[train_index]
            X_test = self.X[test_index]
            y_train = self.y[train_index]
            y_test = self.y[test_index]
            w_train = self.w[train_index]
            logging.info('[Fold %d] %d training samples, %d testing samples',
                         fold, len(X_train), len(X_test))
            logging.info('[Fold %d] Fitting main model...', fold)
            main_model.fit(X_train, y_train, sample_weight=w_train)
            y_train_pred = main_model.predict(X_train)
            y_diff = y_train == y_train_pred
            y_diff = y_diff.astype(np.int)
            logging.info('[Fold %d] Fitting meta model...', fold)
            meta_model.fit(X_train, y_diff)
            y_train_meta = meta_model.predict(X_train)
            print_metrics(y_train, y_train_pred, y_train_meta, 'Fold %d Training ' % (fold,))
            y_test_pred = main_model.predict(X_test)
            y_test_meta = meta_model.predict(X_test)
            accuracy = print_metrics(y_test, y_test_pred, y_test_meta, 'Fold %d Testing ' % (fold,))
            accuracy_table.append([str(fold), '%.2f%%' % (accuracy * 100,)])
            accuracy_sum += accuracy
            fold += 1
        accuracy_table.append(['Average', '%.2f%%' % (accuracy_sum / k * 100)])
        logging.info(utils.get_header('Model Accuracy') + '\n' +
                     tabulate(accuracy_table, headers=['Fold', 'Accuracy'], tablefmt='grid'))

    def _get_model_paths(self, suffix):
        main_model_path = os.path.join(self.root_dir, utils.MODELS_DIR, 'main_%s.p' % (suffix,))
        meta_model_path = os.path.join(self.root_dir, utils.MODELS_DIR, 'meta_%s.p' % (suffix,))
        return main_model_path, meta_model_path

    def train(self):
        main_model = ensemble.RandomForestClassifier(**self.hyper_parameters)
        meta_model = ensemble.RandomForestRegressor(**self.hyper_parameters)
        logging.info('Fitting main model...')
        main_model.fit(self.X, self.y, sample_weight=self.w)
        y_pred = main_model.predict(self.X)
        y_diff = self.y == y_pred
        y_diff = y_diff.astype(np.int)
        logging.info('Fitting meta model...')
        meta_model.fit(self.X, y_diff)
        y_meta = meta_model.predict(self.X)
        accuracy = print_metrics(self.y, y_pred, y_meta, 'Training ')
        logging.info('Accuracy: %.2f%%', accuracy * 100)
        main_model_path, meta_model_path = self._get_model_paths(str((np.round(accuracy*1E4))))
        with open(main_model_path, 'wb') as f_main:
            pickle.dump(main_model, f_main)
        with open(meta_model_path, 'wb') as f_meta:
            pickle.dump(meta_model, f_meta)
        logging.info('Model saved at\n%s\n%s', main_model_path, meta_model_path)

    def evaluate(self):
        main_model_path, meta_model_path = self._get_model_paths(self.model_suffix)
        logging.info('Loading model...')
        with open(main_model_path, 'rb') as f_main:
            main_model = pickle.load(f_main)
        with open(meta_model_path, 'rb') as f_meta:
            meta_model = pickle.load(f_meta)
        logging.info('Predicting...')
        y_pred = main_model.predict(self.X)
        y_meta = meta_model.predict(self.X)
        print_metrics(self.y, y_pred, y_meta, 'Evaluation ')


def main():
    parser = argparse.ArgumentParser(description='Stock trading ML model.')
    parser.add_argument('--model_suffix', default=None,
                        help='Model to load')
    parser.add_argument('--data_files', required=True, nargs='+',
                        help='Data to train on.')
    parser.add_argument('--action', default='dev', choices=['dev', 'train', 'eval'])
    args = parser.parse_args()
    utils.logging_config()
    ml = ML(args.data_files, args.model_suffix)
    if args.action == 'train':
        ml.train()
    elif args.action == 'dev':
        ml.k_fold_cross_validation(5)
    elif args.action == 'eval':
        ml.evaluate()
    else:
        raise ValueError('Invalid action')


if __name__ == '__main__':
    main()

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
    confusion_matrix_main = metrics.confusion_matrix(y_true, y_pred)
    if len(confusion_matrix_main) == 2:
        confusion_table_main = [['', 'Predict Short', 'Predict Long'],
                                ['True Short', confusion_matrix_main[0][0], confusion_matrix_main[0][1]],
                                ['True Long', confusion_matrix_main[1][0], confusion_matrix_main[1][1]]]
        outputs.extend([utils.get_header(title_prefix + 'Main Confusion Matrix'),
                        tabulate(confusion_table_main, tablefmt='grid')])

    indices = sorted(range(len(y_meta)), key=lambda i: y_meta[i], reverse=True)
    indices = indices[:8]
    dropped = len(y_meta) - len(indices)
    outputs.append('Threshold: %.2f. %d (%.1f%%) samples dropped. %d (%.1f%%) samples preserved.' % (
        y_meta[indices[-1]], dropped, dropped / len(y_meta) * 100, len(indices), len(indices) / len(y_meta) * 100))
    y_true_meta = y_true[indices]
    y_pred_meta = y_pred[indices]
    confusion_matrix_final = metrics.confusion_matrix(y_true_meta, y_pred_meta)
    if len(confusion_matrix_final) == 2:
        confusion_table_final = [['', 'Predict Short', 'Predict Long'],
                                 ['True Short', confusion_matrix_final[0][0], confusion_matrix_final[0][1]],
                                 ['True Long', confusion_matrix_final[1][0], confusion_matrix_final[1][1]]]
        outputs.extend([utils.get_header(title_prefix + 'Final Confusion Matrix'),
                        tabulate(confusion_table_final, tablefmt='grid')])

    accuracy_main = metrics.accuracy_score(y_true, y_pred)
    accuracy_final = metrics.accuracy_score(y_true_meta, y_pred_meta)
    benchmark = np.sum(y_true == 1) / len(y_true)
    benchmark = benchmark if benchmark > 0.5 else 1 - benchmark
    accuracy_table = [['Accuracy Main', '%.2f%%' % (accuracy_main * 100,)],
                      ['Accuracy Final', '%.2f%%' % (accuracy_final * 100,)],
                      ['Benchmark', '%.2f%%' % (benchmark * 100,)]]
    outputs.extend([utils.get_header(title_prefix + 'Accuracy'),
                    tabulate(accuracy_table, tablefmt='grid')])

    logging.info('\n'.join(outputs))
    return accuracy_final


def process_data(df):
    X, y, w = [], [], []
    logging.info('Processing data...')
    for _, row in df.iterrows():
        gain = row['Gain']
        if gain >= 0:
            y_value = 1
        else:
            y_value = 0
        x_value = [row[col] for col in utils.ML_FEATURES]
        X.append(x_value)
        y.append(y_value)
        w.append(np.abs(gain))
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    logging.info('%d data samples loaded', len(X))
    return X, y, w


class ML(object):

    def __init__(self, data_files, start_date=None, end_date=None, model_suffix=None):
        data_files = data_files
        self.model_suffix = model_suffix
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        logging.info('Reading csv data...')
        self.df = pd.concat([pd.read_csv(data_file) for data_file in data_files])
        self.df.dropna(inplace=True)
        indices = []
        for i, row in self.df.iterrows():
            if (not start_date or row['Date'] >= start_date) and (not end_date or row['Date'] <= end_date):
                indices.append(i)
        self.df = self.df.iloc[indices]
        self.hyper_parameters = {'max_depth': 2,
                                 'min_samples_leaf': 0.1,
                                 'n_jobs': 2}
        logging.info('Model hyper-parameters: %s', self.hyper_parameters)

    def k_fold_cross_validation(self, k):
        X, y, w = process_data(self.df)
        main_model = ensemble.RandomForestClassifier(**self.hyper_parameters)
        meta_model = ensemble.RandomForestRegressor(**self.hyper_parameters)
        k_fold = KFold(n_splits=k, shuffle=True, random_state=0)
        fold = 1
        accuracy_sum = 0
        accuracy_table = []
        for train_index, test_index in k_fold.split(X):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            w_train = w[train_index]
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

    def train(self, X=None, y=None, w=None, save_model=False):
        if X is None:
            X, y, w = process_data(self.df)
        main_model = ensemble.RandomForestClassifier(**self.hyper_parameters)
        meta_model = ensemble.RandomForestRegressor(**self.hyper_parameters)
        logging.info('Fitting main model...')
        main_model.fit(X, y, sample_weight=w)
        y_pred = main_model.predict(X)
        y_diff = y == y_pred
        y_diff = y_diff.astype(np.int)
        logging.info('Fitting meta model...')
        meta_model.fit(X, y_diff)
        if save_model:
            y_meta = meta_model.predict(X)
            accuracy = print_metrics(y, y_pred, y_meta, 'Training ')
            main_model_path, meta_model_path = self._get_model_paths(str(int(round(accuracy*1E4))))
            with open(main_model_path, 'wb') as f_main:
                pickle.dump(main_model, f_main)
            with open(meta_model_path, 'wb') as f_meta:
                pickle.dump(meta_model, f_meta)
            logging.info('Model saved at\n%s\n%s', main_model_path, meta_model_path)
        return main_model, meta_model

    def evaluate(self):
        X, y, _ = process_data(self.df)
        main_model_path, meta_model_path = self._get_model_paths(self.model_suffix)
        logging.info('Loading model...')
        with open(main_model_path, 'rb') as f_main:
            main_model = pickle.load(f_main)
        with open(meta_model_path, 'rb') as f_meta:
            meta_model = pickle.load(f_meta)
        logging.info('Predicting...')
        y_pred = main_model.predict(X)
        y_meta = meta_model.predict(X)
        print_metrics(y, y_pred, y_meta, 'Evaluation ')

    def continuous_training(self, training_days, testing_days, step_days):
        dates = self.df['Date'].unique()
        accuracy_table = []
        accuracy_sum = 0
        accuracy_count = 0
        for i_date in range(0, len(dates), step_days):
            start_date = pd.to_datetime(dates[i_date])
            train_indices = []
            test_indices = []
            current_date = None
            day_count = 0
            for i, row in self.df.iterrows():
                date = pd.to_datetime(row['Date'])
                if date < start_date:
                    continue
                if current_date != date:
                    current_date = date
                    day_count += 1
                if day_count <= training_days:
                    train_indices.append(i)
                elif day_count <= training_days + testing_days:
                    test_indices.append(i)
                else:
                    X_train, y_train, w_train = process_data(self.df.iloc[train_indices])
                    X_test, y_test, _ = process_data(self.df.iloc[test_indices])
                    main_model, meta_model = self.train(X_train, y_train, w_train)
                    y_pred = main_model.predict(X_test)
                    y_meta = meta_model.predict(X_test)
                    test_range = '%s ~ %s' % (self.df.iloc[test_indices[0]]['Date'],
                                              self.df.iloc[test_indices[-1]]['Date'])
                    accuracy = print_metrics(y_test, y_pred, y_meta, 'Evaluation %s ' % (test_range,))
                    accuracy_table.append([test_range, '%.2f%%' % (accuracy * 100,)])
                    accuracy_sum += accuracy
                    accuracy_count += 1
                    logging.info('Average accuracy: %.2f%%', accuracy_sum / accuracy_count * 100)
                    break
            else:
                break
        if accuracy_count:
            accuracy_table.append(['Average', '%.2f%%' % (accuracy_sum / accuracy_count * 100)])
            logging.info(utils.get_header('Model Accuracy') + '\n' +
                         tabulate(accuracy_table, headers=['Date', 'Accuracy'], tablefmt='grid'))
        else:
            logging.info('No training performed')


def main():
    parser = argparse.ArgumentParser(description='Stock trading ML model.')
    parser.add_argument('--model_suffix', default=None,
                        help='Model to load')
    parser.add_argument('--data_files', required=True, nargs='+',
                        help='Data to train on.')
    parser.add_argument('--start_date', default=None,
                        help='Start date of the data.')
    parser.add_argument('--end_date', default=None,
                        help='End date of the data.')
    parser.add_argument('--action', default='dev', choices=['dev', 'train', 'eval', 'cont'])
    args = parser.parse_args()
    utils.logging_config()
    ml = ML(args.data_files, args.start_date, args.end_date, args.model_suffix)
    if args.action == 'train':
        ml.train(save_model=True)
    elif args.action == 'dev':
        ml.k_fold_cross_validation(5)
    elif args.action == 'eval':
        ml.evaluate()
    elif args.action == 'cont':
        ml.continuous_training(20, 1, 1)
    else:
        raise ValueError('Invalid action')


if __name__ == '__main__':
    main()

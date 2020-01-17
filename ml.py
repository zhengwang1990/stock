import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from common import *


def read_df():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  df = pd.read_csv(os.path.join(dir_path, OUTPUTS_DIR, 'simulate_stats.csv'))
  return df


def plot_data():
  df = read_df()
  y = df.get('Gain')
  for key, _ in df.iteritems():
    if key not in ('Gain', 'Symbol', 'Date'):
      x = df.get(key)
      plt.figure()
      plt.plot(x, y, 'o', markersize=3)
      plt.plot([np.min(x), np.max(x)],[0, 0], '--')
      plt.title(key + ' v.s. Gain')
      plt.show()


def load_data():
  df = read_df()
  keys = [key for key, _ in df.iteritems() if key not in ('Gain', 'Symbol', 'Date')]
  x, y = [], []
  for row in df.itertuples():
    if row.Threshold <= 0:
      continue
    x_value = [getattr(row, key) for key in keys]
    x_value.extend([row.Day_Range_Change - row.Threshold,
                    row.Day_Range_Change / row.Threshold])
    y_value = row.Gain / 5 if np.abs(row.Gain) < 5 else np.sign(row.Gain)
    y_value = (y_value + 1) / 2
    x.append(x_value)
    y.append(y_value)
  x = np.array(x)
  y = np.array(y)
  return x, y


def get_model():
  df = read_df()
  xdim = len(df.columns) - 1
  model = keras.Sequential([
      keras.layers.Input(shape=(xdim,)),
      keras.layers.Dense(20, activation='tanh',
                         input_shape=(xdim,),
                         kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
      keras.layers.Dense(40, activation='tanh'),
      keras.layers.Dense(20, activation='tanh'),
      keras.layers.Dense(20, activation='tanh'),
      keras.layers.Dense(20, activation='tanh'),
      keras.layers.Dense(5, activation='tanh'),
      keras.layers.Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  model.summary()
  return model


def train_model(x, y, model):
  earlystopping = keras.callbacks.EarlyStopping(
      monitor='loss', patience=5, restore_best_weights=True)
  model.fit(x, y, epochs=100, callbacks=[earlystopping])
  model.save('model.hdf5')


def load_model():
  model = keras.models.load_model('model.hdf5')
  return model


def predict(x, y, model):
  pred = model.predict(x)
  print(pred)
  plt.figure()
  plt.plot(pred, y, 'o', markersize=3)
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()


def main():
  x, y = load_data()
  model = get_model()
  train_model(x, y, model)
  #model = load_model()
  predict(x, y, model)



if __name__ == '__main__':
  main()

import os
import time

import joblib
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide all the tensorflow screen verbosity (unless error)


# create dataset from .csv: open of day+0 + open/close/high/low/volume of day-1

def add_original_features(df, df_new):
    df_new['open'] = df['open']
    df_new['open_1'] = df['open'].shift(1)
    df_new['close_1'] = df['close'].shift(1)
    df_new['high_1'] = df['high'].shift(1)
    df_new['low_1'] = df['low'].shift(1)
    df_new['volume_1'] = df['Volume BTC'].shift(1)


# adding average features

def add_avg_price(df, df_new):
    df_new['avg_price_7'] = df['close'].rolling(7).mean().shift(1)
    df_new['avg_price_30'] = df['close'].rolling(30).mean().shift(1)
    df_new['avg_price_365'] = df['close'].rolling(365).mean().shift(1)
    df_new['ratio_avg_price_7_30'] = df_new['avg_price_7'] / df_new['avg_price_30']
    df_new['ratio_avg_price_7_365'] = df_new['avg_price_7'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']


# adding average volumes features

def add_avg_volume(df, df_new):
    df_new['avg_volume_7'] = df['Volume BTC'].rolling(7).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume BTC'].rolling(30).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume BTC'].rolling(365).mean().shift(1)
    df_new['ratio_avg_volume_7_30'] = df_new['avg_volume_7'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_7_365'] = df_new['avg_volume_7'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']


# adding close price standard deviation features

def add_std_price(df, df_new):
    df_new['std_price_7'] = df['close'].rolling(7).std().shift(1)
    df_new['std_price_30'] = df['close'].rolling(30).std().shift(1)
    df_new['std_price_365'] = df['close'].rolling(365).std().shift(1)
    df_new['ratio_std_price_7_30'] = df_new['std_price_7'] / df_new['std_price_30']
    df_new['ratio_std_price_7_365'] = df_new['std_price_7'] / df_new['std_price_365']
    df_new['ratio-std-price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']


# adding volume standard deviation features

def add_std_volume(df, df_new):
    df_new['std_volume_7'] = df['Volume BTC'].rolling(7).std().shift(1)
    df_new['std_volume_30'] = df['Volume BTC'].rolling(30).std().shift(1)
    df_new['std_volume_365'] = df['Volume BTC'].rolling(365).std().shift(1)
    df_new['ratio_std_volume_7_30'] = df_new['std_volume_7'] / df_new['std_volume_30']
    df_new['ratio_std_volume_7_365'] = df_new['std_volume_7'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']


# adding return based features

def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1).shift(1))
    df_new['return_7'] = ((df['close'] - df['close'].shift(7)) / df['close'].shift(7).shift(1))
    df_new['return_30'] = ((df['close'] - df['close'].shift(30)) / df['close'].shift(30).shift(1))
    df_new['return_365'] = ((df['close'] - df['close'].shift(365)) / df['close'].shift(365).shift(1))
    df_new['moving_avg_7'] = df_new['return_1'].rolling(7).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(30).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(365).mean().shift(1)


# add optional nasdaq features

def add_nasdaq(df, df_new):
    df_new['open_nasdaq'] = df['open_nasdaq']
    df_new['open_nasdaq_1'] = df['open_nasdaq'].shift(1)
    df_new['close_nasdaq_1'] = df['close_nasdaq'].shift(1)
    df_new['high_nasdaq_1'] = df['high_nasdaq'].shift(1)
    df_new['low_nasdaq_1'] = df['low_nasdaq'].shift(1)
    df_new['volume_nasdaq_1'] = df['volume_nasdaq'].shift(1)
    df_new['nd_std_price_7'] = df['close_nasdaq'].rolling(7).std().shift(1)
    df_new['nd_std_price_30'] = df['close_nasdaq'].rolling(30).std().shift(1)
    df_new['nd_std_price_365'] = df['close_nasdaq'].rolling(365).std().shift(1)


# generate all features

def generate_features(df):
    df_new = pd.DataFrame()
    add_original_features(df, df_new)
    add_avg_price(df, df_new)
    add_avg_volume(df, df_new)
    add_std_price(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    # add_nasdaq(df,df_new)
    df_new['close'] = df['close']
    df_new = df_new.dropna(axis=0)
    return df_new


print('Loading .csv file... ', end='')
data_raw = pd.read_csv('Bitstamp_BTCUSD_adapted.csv', index_col='date')
print('done')
data = generate_features(data_raw)
pd.set_option('max_columns', 15)
pd.set_option('display.width', 200)
print(data.round(decimals=3).tail(5))

start_train = '2014-11-28'
stop_train = '2020-12-31'
start_test = '2021-01-01'
stop_test = '2021-10-06'

data_train = data.loc[start_train:stop_train]
X_train = data_train.drop('close', axis=1).values
Y_train = data_train['close'].values
print('\nRaw training data (samples, features): ', end='')
print(data_train.shape)
print('Training data X_train: ', end='')
print(X_train.shape)
print('Training data Y_train: ', end='')
print(Y_train.shape)

data_test = data.loc[start_test:stop_test]
X_test = data_test.drop('close', axis=1).values
Y_test = data_test['close'].values
print('Test data: ', end='')
print(X_test.shape)
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

print('\nStarting Support Vector Regression (SVR) calculation... ', end='')
start_time = int(time.time())
param_grid = [{'kernel': ['linear'], 'C': [100, 300, 500], 'epsilon': [0.00003, 0.0001]},
              {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [10, 100, 1000], 'epsilon': [0.00003, 0.0001]}]
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_scaled_train, Y_train)
print('done')
stop_time = int(time.time())
print('\nPerformance of SVR: (' + str(stop_time - start_time) + ' sec)')
print(grid_search.best_params_)
svr_best = grid_search.best_estimator_
predictions_svr = svr_best.predict(X_scaled_test)
print(f'MSE: {mean_squared_error(Y_test, predictions_svr):.3f}')
print(f'MAE: {mean_absolute_error(Y_test, predictions_svr):.3f}')
print(f'R^2: {r2_score(Y_test, predictions_svr):.3f}')
svr_filename = 'bitcoin_model-svr.pkl'
joblib.dump(svr_best, svr_filename)

# Show simple result for 1 day

print('\nSimple prediction for ' + stop_test + ':')
print('Actual open = ', end='')
print(X_test[:, 0][-1])
print('Actual close = ', end='')
print(Y_test[-1])
print('Predicted close (svr) = ', end='')
print(predictions_svr[-1].round(decimals=0))


def do_simulation(prediction, x_testdata, wallet_euro, wallet_btc):
    current_euro = wallet_euro
    current_btc = wallet_btc
    fee = 0.005
    number_of_transactions = 0
    for teller, pred in enumerate(prediction):
        current_btc_value = x_testdata[:, 0][teller]
        if pred > current_btc_value * (1.0 + 3 * fee):
            # buy or hold
            if current_euro > 0:
                # buy
                # print('buy ', end='')
                current_btc += (1.0 - fee) * current_euro / current_btc_value
                current_euro = 0
                number_of_transactions += 1
        else:
            if pred < current_btc_value * (1.0 - 2 * fee):
                # sell or hold
                if current_btc > 0:
                    # sell
                    # print('sell ', end='')
                    current_euro += (1.0 - fee) * current_btc * current_btc_value
                    current_btc = 0
                    number_of_transactions += 1
    return current_euro, current_btc, number_of_transactions


start_amount_euro = 25000
start_amount_btc = 0.5
wallet = do_simulation(predictions_svr, X_test, start_amount_euro, start_amount_btc)
print('\nSvr trading simulation result: ', end='')
print(wallet)
start_amount_in_euro = start_amount_euro + start_amount_btc * X_test[:, 0][0]
start_amount_in_btc = start_amount_euro / X_test[:, 0][0] + start_amount_btc
final_amount_in_euro = wallet[0] + wallet[1] * X_test[:, 0][-1]
final_amount_in_btc = wallet[0] / X_test[:, 0][-1] + wallet[1]
print('We started with ' + str(int(start_amount_in_euro)) + ' € or ' + str(start_amount_in_btc.round(3)) + ' BTC')
print('We ended up with ' + str(int(final_amount_in_euro)) + ' € or ' + str(final_amount_in_btc.round(3))
      + ' BTC and executed ' + str(wallet[2]) + ' transactions')

# Plotting results

plt.plot(data_test.index, Y_test, c='k')
plt.plot(data_test.index, predictions_svr, c='b')
plt.xticks(range(0, 252, 10), rotation=60)
plt.xlabel('Date')
plt.ylabel('Close price')
plt.legend(['Truth', 'SVR base'])
plt.show()

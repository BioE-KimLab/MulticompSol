import re
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics


def calc_mae_rmse_r2(df, true_val_col, predicted_val_col):
    mae = np.abs(df[true_val_col] - df[predicted_val_col]).mean()
    rmse = sklearn.metrics.mean_squared_error(y_true=df[true_val_col], y_pred=df[predicted_val_col], squared=False) # Will fail in scikit-learn > 1.3
    r2 = sklearn.metrics.r2_score(y_true=df[true_val_col],
                                        y_pred=df[predicted_val_col])
    return mae,rmse,r2

def calc_tvtest_stats(data, target, predicted_val_col='predicted'):
    train_data = data.loc[data['Train_Valid_Test'].str.fullmatch('Train'), :]
    valid_data = data.loc[data['Train_Valid_Test'].str.fullmatch('Valid'), :]
    test_data = data.loc[data['Train_Valid_Test'].str.fullmatch('Test'), :]


    if train_data.shape[0] == 0:
        mae_train = np.nan
        rmse_train = np.nan
        r2_train = np.nan
    else:
        mae_train, rmse_train, r2_train = calc_mae_rmse_r2(train_data,
                                                           true_val_col=target,
                                                           predicted_val_col= predicted_val_col)

    if valid_data.shape[0] == 0:
        mae_valid = np.nan
        rmse_valid = np.nan
        r2_valid = np.nan
    else:
        mae_valid, rmse_valid, r2_valid = calc_mae_rmse_r2(valid_data,
                                                           true_val_col=target,
                                                           predicted_val_col=predicted_val_col)

    if test_data.shape[0] == 0:
        mae_test = np.nan
        rmse_test = np.nan
        r2_test = np.nan
    else:
        mae_test, rmse_test, r2_test = calc_mae_rmse_r2(test_data,
                                                           true_val_col=target,
                                                           predicted_val_col=predicted_val_col)



    mae_vals = [mae_train, mae_valid, mae_test]
    rmse_vals = [rmse_train, rmse_valid, rmse_test]
    r2_vals = [r2_train, r2_valid, r2_test]
    return mae_vals, rmse_vals, r2_vals

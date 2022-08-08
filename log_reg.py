import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from help_code_demo import stats_report
import numpy as np
import pandas as pd


def log_reg(data_all):
    """
    Build logistic regression model
    :param data_all: data in dataframe
    :return: model
    """
    X_train = np.array([data for data in data_all.loc[data_all['Mode'] == 'train']['Data']])
    y_train = data_all.loc[data_all['Mode'] == 'train']['Class'].to_numpy(dtype=float)
    print(X_train.shape)
    print(y_train.shape)
    X_test = np.array([data for data in data_all.loc[data_all['Mode'] == 'test']['Data']])
    y_test = data_all.loc[data_all['Mode'] == 'test']['Class'].to_numpy()
    print(X_test.shape)
    print(y_test.shape)

    print("Logistic Regression model training and testing")
    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    filename = 'log_reg_model.pkl'
    pickle.dump(model, open('./saved_models/' + filename, 'wb'))
    print("Logistic Regression model training and saving complete")
    total = 5625
    y_pred = model.predict(X_test)
    print(np.sum(y_pred == 1))

    correct = (y_pred == y_test).sum()
    print('test acc = ', correct / total)
    print('test correct = ', correct)
    print('train acc = ', model.score(X_train, y_train))

    segs_TN, segs_FN, segs_FP, segs_TP = confusion_matrix(y_test, y_pred).reshape(-1)

    # report metrics
    stats_file = open('./records/' + 'seg_stat_log_reg.txt', 'w')
    stats_file.write('segments: TP, FN, FP, TN\n')
    output_segs = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
    stats_file.write(output_segs + '\n')

    return model

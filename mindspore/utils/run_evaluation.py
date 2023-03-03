from scipy.stats import kendalltau
from sklearn import model_selection
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def calculate_MSE(y_pred, y_test):
    """
    Calculate MSE between y_pred and y_test
    :param y_pred: predict values
    :param y_test: groud truth
    :return: mse
    """
    mse_list = np.array([(element_x - element_y) ** 2 for element_x, element_y in zip(y_pred, y_test)])
    mse = np.mean(mse_list)
    return mse


def evaluateResult(y_pred, y_test, method):
    """
    Calculate kTau, r2score and draw two chart to show the evaluation result.
    :param y_pred: predict values
    :param y_test: groud truth
    :param method: model that the prediction used
    :return: Nan
    """
    score = r2_score(y_test, y_pred)
    result_arg = np.argsort(y_pred)
    y_test_arg = np.argsort(y_test)
    result_rank = np.zeros(len(result_arg))
    y_test_rank = np.zeros(len(y_test_arg))
    for i in range(len(y_test_arg)):
        result_rank[result_arg[i]] = i
        y_test_rank[y_test_arg[i]] = i
    KTau, _ = kendalltau(result_rank, y_test_rank)
    print(
        'Method: {:}, KTau: {:}, MSE: {:}, R2score: {:}'.format(method, KTau, calculate_MSE(y_test, y_pred),
                                                                score))
    if True:
        x = np.arange(0, 1, 0.01)
        y = x
        plt.figure(figsize=(5, 5))
        plt.plot(x, y, 'g', label='y_test = result')
        plt.scatter(y_pred, y_test, s=1)
        plt.xlabel("predict_result")
        plt.ylabel("y_test")
        plt.title(f"Method:{method}---score:{score}")
        plt.legend(loc="best")
        plt.show()

        x = np.arange(0, len(y_test), 0.1)
        y = x
        plt.figure(figsize=(6, 6))
        line_color = '#1F77D0'
        plt.plot(x, y, c=line_color, linewidth=1)
        point_color = '#FF4400'
        plt.scatter(result_rank, y_test_rank, c=point_color, s=2)
        plt.xlabel("predict_result")
        plt.ylabel("y_test")
        plt.title(f"Method:{method}---KTau:{KTau}")
        plt.xlim(xmax=y_pred.shape[0], xmin=0)
        plt.ylim(ymax=y_pred.shape[0], ymin=0)
        plt.show()


def run_model(model, vector_size=166, dataset_path='data/dataset.csv'):
    """
    Train model and predict accuracy of CNNs.
    :param model: model for train and prediction
    :param vector_size: size of onehot vector
    :param dataset_path: file path of the dataset
    :returns: predict values; test ground truth
    """
    dataset = pd.read_csv(dataset_path)

    newAttrName = [str(i) for i in range(0, vector_size)]
    X = np.array(dataset[newAttrName])
    y = np.array(dataset['y'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, test_size=0.3,
                                                                        random_state=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred, y_test

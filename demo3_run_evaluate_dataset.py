from sklearn import svm, neighbors, ensemble, tree
from sklearn.linear_model import LinearRegression
from utils.run_evaluation import run_model, evaluateResult

if __name__ == '__main__':

    vector_size = 166

    # 1.decision tree regression
    model_decision_tree_regression = tree.DecisionTreeRegressor()

    # 2.linear regression
    model_linear_regression = LinearRegression()

    # 3.SVM regression
    model_svm = svm.SVR()

    # 4.kNN regression
    model_k_neighbor = neighbors.KNeighborsRegressor()

    # 5.random forest regression
    model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=100)

    # 6.GBRT regression
    model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor()

    modelDict = {'Tree': model_decision_tree_regression, 'Linear': model_linear_regression, 'KNN': model_k_neighbor,
                 'RandomForest': model_random_forest_regressor,
                 'GBRT': model_gradient_boosting_regressor}

    for i in modelDict.keys():

        y_pred, y_test = run_model(model=modelDict[i], vector_size=vector_size)

        evaluateResult(y_pred, y_test, i)

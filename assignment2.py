import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from datetime import datetime
import time
import shap

from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

target_encoder = None


def get_col_types(data):
    features_num = len(data.columns)
    nominal_cols = set()
    nominal_cols_indexes = set()
    for index in range(83):
        column_name = 'A' + str(index + 1)
        if column_name in data.columns:
            nominal_cols.add(column_name)
            nominal_cols_indexes.add(index)
    binary_cols = set(
        [col for col in data.loc[:, data.columns != 'CLASS'] if np.isin(data[col].dropna().unique(), [0, 1]).all()])
    numeric_cols = set()
    union_set = nominal_cols.union(binary_cols)
    for col_name in data.columns:
        if col_name not in union_set and col_name != 'CLASS':
            numeric_cols.add(col_name)
    return nominal_cols, nominal_cols_indexes, binary_cols, numeric_cols


def fill_missing_data(data, nominal_cols, binary_cols, numeric_cols, dist_dict):
    union_set = nominal_cols.union(binary_cols)
    for column_name in union_set:
        # data[column_name] = data[column_name].fillna(data[column_name].mode().iloc[0])
        if column_name not in dist_dict.keys():
            distribution = data[column_name].fillna(data[column_name].mode().iloc[0])
            # distribution = data[column_name].dropna().value_counts(normalize=True)
            dist_dict[column_name] = distribution
        else:
            distribution = dist_dict[column_name]
        data[column_name].fillna(distribution, inplace=True)
        # missing = data[column_name].isnull()
        # data.loc[missing, column_name] = np.random.choice(distribution.index, size=len(data[missing]), p=distribution.values)

    for column_name in numeric_cols:
        if column_name not in dist_dict.keys():
            average = data[column_name].mean()
            dist_dict[column_name] = average
        else:
            average = dist_dict[column_name]
        data[column_name].fillna(average, inplace=True)


def transform_categorical_columns(data, nominal_cols, dummies=False, labelencoder=False):
    class_encoder = LabelEncoder()
    global target_encoder
    fit_trans = False
    if target_encoder is None:
        fit_trans = True
        target_encoder = TargetEncoder(cols=nominal_cols)

    if dummies:
        new_Data = pd.get_dummies(data.iloc[:, data.columns != "CLASS"], columns=nominal_cols)
    else:
        new_Data = data

    if labelencoder:
        if 'CLASS' in data.columns:
            class_col = pd.Series(class_encoder.fit_transform(new_Data["CLASS"]))
            del new_Data['CLASS']
        # for column_name in nominal_cols:
        if fit_trans:
            new_Data = target_encoder.fit_transform(new_Data, class_col)
            new_Data['CLASS'] = class_col
        else:
            new_Data = target_encoder.transform(new_Data)
    else:
        if 'CLASS' in data.columns:
            new_Data['CLASS'] = pd.Series(class_encoder.fit_transform(new_Data["CLASS"]))
    return new_Data


def read_data(filename, dist_dict, dummies=False, labelencoder=False):
    path = 'data/'
    data = pd.read_csv(path + filename)
    if len(dist_dict) > 0:
        for column_name in data.columns:
            if column_name not in dist_dict.keys():
                del data[column_name]
    else:
        # data = data.loc[:, data.isnull().mean() < 0.8]
        data = data.dropna(how='all', axis=1)  # remove columns that don't have any data in them.
    nominal_cols, nominal_cols_indexes, binary_cols, numeric_cols = get_col_types(data)
    fill_missing_data(data, nominal_cols, binary_cols, numeric_cols, dist_dict)
    data = transform_categorical_columns(data, nominal_cols, dummies, labelencoder)
    data.to_csv(path + 'edited_' + filename, sep=",")
    return data, nominal_cols_indexes


def write_prediction(pred):
    current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    with open('results/Predictions/' + 'test_pred_' + current_time + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['Id'] + ['ProbToYes'])
        for index in range(len(pred)):
            spamwriter.writerow([index + 1] + [pred[index]])


def calculate_auc(pred, actual):
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def draw_roc(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('results/' + 'roc_curve.png')
    plt.close()


def calculateAUC(label, classifier, X, Y):
    pred = classifier.predict(X)
    print(label + ':')
    print('\tScore: ' + str(classifier.score(X, Y)))
    print('\tAUC: ' + str(roc_auc_score(Y, pred)))
    print('\tAUC PROB: ' + str(roc_auc_score(Y, classifier.predict_proba(X)[:, 1])))
    draw_roc(Y, classifier.predict_proba(X)[:, 1])
    return pred


def printSHAP(trained_model, data, X, list_to_plot):
    # summary plots
    shap_values = shap.TreeExplainer(trained_model).shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig('results/Shap/' + 'shap_summary_bar.png')
    plt.close()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('results/Shap/' + 'shap_summary.png')
    plt.close()

    shap.initjs()

    # importance
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    importance_df.to_csv('results/Shap/' + 'features_importance.csv')

    # depentence contribution plots
    features_list = list(importance_df['column_name'][:5].values)
    for feature in features_list:
        shap.dependence_plot(feature, shap_values, X, interaction_index=None, show=False)
        plt.savefig('results/Shap/Dependence_Contribution/' + 'Dependence_Contribution_' +
                    feature + '.png')
        plt.close()

    # explainer plots for single records
    explainerModel = shap.TreeExplainer(trained_model)
    shap_values_Model = explainerModel.shap_values(data)
    for j in list_to_plot:
        shap.force_plot(explainerModel.expected_value, shap_values_Model[j], train_data.iloc[[j]], show=False,
                        matplotlib=True)
        plt.savefig('results/Shap/' + 'shap_explainer_' + str(j) + '.png')
        plt.close()


gradientBoostingClassifier = GradientBoostingClassifier(n_estimators=3000, max_leaf_nodes=4, max_depth=None,
                                                        random_state=2,
                                                        min_samples_split=200, learning_rate=0.01, subsample=0.5)
xgbClassifier = xgb.XGBClassifier(n_estimators=20000, max_leaf_nodes=4, min_samples_split=500, learning_rate=0.001,
                                  subsample=0.5, verbosity=0,
                                  max_depth=3)  # lr=0.001, n_estimators = 20000, max depth=3: avg = 0.735

randomForestClassifier = RandomForestClassifier(verbose=0, n_estimators=900, max_depth=15, n_jobs=20,
                                                min_samples_split=500)  # avg = 0.675
extraTreesClassifier = ExtraTreesClassifier(verbose=0, n_estimators=1000, max_depth=10, n_jobs=20,
                                            min_samples_split=100)  # avg = 0.658
decisionTreeClassifier = DecisionTreeClassifier(min_samples_split=500)  # avg = 0.654
adaBoostClassifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), algorithm="SAMME",
                                        n_estimators=300)  # avg = 0.715

parameters = {
    "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

xgbClassifierGrid = GridSearchCV(xgb.XGBClassifier(),
                                 parameters, n_jobs=4,
                                 scoring="neg_log_loss",
                                 cv=3)

catBoostClassifier = CatBoostClassifier(learning_rate=0.05, subsample=0.5, verbose=False)  # avg = 0.7497

lgbClassifier = lgb.LGBMClassifier(learning_rate=0.05, subsample=0.5)  # avg = 0.7445

svc = SVC(kernel='rbf', class_weight='balanced', gamma=0.01, C=1e3, probability=True)  # avg = 0.413

estimators = [
    ('rf', randomForestClassifier),
    ('xgb', xgbClassifier),
    ('gbm', gradientBoostingClassifier),
    ('catbm', catBoostClassifier),
    ('adab', adaBoostClassifier)
]

stackingClassifier = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)  # avg = 0.741

logisticRegression = LogisticRegression()

classifier = catBoostClassifier
do_cross_val = False
use_all_data_for_end_classifing = True
saving_result = True
num_of_folds = 5

dist_dict = {}
if classifier is catBoostClassifier:
    train_data, nominal_cols_indexes = read_data('train.CSV', dist_dict)
    test_data, _ = read_data('test.CSV', dist_dict)
else:
    train_data, nominal_cols_indexes = read_data('train.CSV', dist_dict, labelencoder=True)
    test_data, _ = read_data('test.CSV', dist_dict, labelencoder=True)

if use_all_data_for_end_classifing:
    X_train = train_data.iloc[:, train_data.columns != 'CLASS']
    Y_train = train_data["CLASS"]
else:
    X_train, X_validate, Y_train, Y_validate = train_test_split(train_data.iloc[:, train_data.columns != 'CLASS'],
                                                                train_data["CLASS"], test_size=0.3, random_state=42)

if do_cross_val:
    print('Doing cross val')
    start_time = time.time()
    scores = cross_val_score(classifier, X_train, Y_train, cv=num_of_folds, scoring='roc_auc')
    print(str(num_of_folds) + ' - fold scores:')
    print(scores)
    print('avg = ' + str(np.mean(scores)))
    print("time for " + str(num_of_folds) + "-fold validation: %s seconds ---" % (time.time() - start_time))
else:
    print('Not doing cross val')

if saving_result:
    start_time = time.time()
    if classifier is catBoostClassifier:
        classifier.fit(X_train, Y_train, nominal_cols_indexes)
    else:
        classifier.fit(X_train, Y_train)

    print("time for fitting: %s seconds ---" % (time.time() - start_time))
    calculateAUC('Train', classifier, X_train, Y_train)
    if not use_all_data_for_end_classifing:
        calculateAUC('Validate', classifier, X_validate, Y_validate)
    test_pred = classifier.predict_proba(test_data)[:, 1]
    write_prediction(test_pred)
    printSHAP(classifier, train_data, X_train, [0, 5, 10])

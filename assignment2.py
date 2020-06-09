import pandas as pd
import matplotlib as plt
import csv
import numpy as np
from datetime import datetime

from IPython.core.display import display
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
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
# from deepstack.ensemble import StackEnsemble
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import time

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
            distribution = data[column_name].dropna().value_counts(normalize=True)
            dist_dict[column_name] = distribution
        else:
            distribution = dist_dict[column_name]
        missing = data[column_name].isnull()
        data.loc[missing, column_name] = np.random.choice(distribution.index, size=len(data[missing]),
                                                          p=distribution.values)

    for column_name in numeric_cols:
        if column_name not in dist_dict.keys():
            average = data[column_name].mean()
            dist_dict[column_name] = average
        else:
            average = dist_dict[column_name]
        data[column_name].fillna((average), inplace=True)


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
    with open('results/test_pred_' + current_time + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['Id'] + ['ProbToYes'])
        for index in range(len(pred)):
            spamwriter.writerow([index + 1] + [pred[index]])


def calculate_auc(pred, actual):
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def calculateAUC(label, classifier, X, Y):
    pred = classifier.predict(X)
    print(label + ':')
    print('\tScore: ' + str(classifier.score(X, Y)))
    print('\tAUC: ' + str(roc_auc_score(Y, pred)))
    print('\tAUC PROB: ' + str(roc_auc_score(Y, classifier.predict_proba(X)[:, 1])))
    return pred


def printSHAP(trained_model, records, list_to_plot):
    import shap  # package used to calculate Shap values

    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(trained_model)

    # Calculate Shap values
    shap.initjs()
    for i in list_to_plot:
        shap_values = explainer.shap_values(records.iloc[i])
        shap.force_plot(round(explainer.expected_value[1], 3), shap_values[1], records.iloc[i].round(3), show=False, matplotlib=True)
        plt.pyplot.savefig('shap_plot_' + str(i) + '.png')


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

'''
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

xgbClassifier = GridSearchCV(xgb.XGBClassifier(),
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)
'''

catBoostClassifier = CatBoostClassifier(iterations=10000, learning_rate=0.01, depth=2, subsample=0.5, verbose=False)

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

# classifier = xgb.XGBClassifier(n_estimators= 2000, max_leaf_nodes= 4, random_state= 2, min_samples_split= 500, learning_rate= 0.03, subsample= 0.5)
classifier = decisionTreeClassifier
do_cross_val = True
saving_result = True
num_of_folds = 2

dist_dict = {}
if False:  # classifier is catBoostClassifier:
    train_data, nominal_cols_indexes = read_data('train.CSV', dist_dict)
    test_data, _ = read_data('test.CSV', dist_dict)
else:
    train_data, nominal_cols_indexes = read_data('train.CSV', dist_dict, labelencoder=True)
    test_data, _ = read_data('test.CSV', dist_dict, labelencoder=True)

if do_cross_val:
    X_train = train_data.iloc[:, train_data.columns != 'CLASS']
    Y_train = train_data["CLASS"]
else:
    X_train, X_validate, Y_train, Y_validate = train_test_split(train_data.iloc[:, train_data.columns != 'CLASS'],
                                                                train_data["CLASS"], test_size=0.3, random_state=42)

'''
train_data_lgb = lgb.Dataset(X_train, label=Y_train, categorical_feature=nominal_cols_indexes)
test_data_lgb = lgb.Dataset(X_validate, label=Y_validate)


#
# Train the model
#

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.001,
    'n_estimators': 10000,
    'verbose': 0
}

model = lgb.train(parameters,
                       train_data_lgb,
                       valid_sets=test_data_lgb,
                       num_boost_round=50000,
                       early_stopping_rounds=10000)

test_pred = model.predict(test_data)
write_prediction(test_pred)

'''

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
    if False:  # classifier is catBoostClassifier:
        classifier.fit(X_train, Y_train, nominal_cols_indexes)
    else:
        classifier.fit(X_train, Y_train)

    print("time for fitting: %s seconds ---" % (time.time() - start_time))
    calculateAUC('Train', classifier, X_train, Y_train)
    if not do_cross_val:
        calculateAUC('Validate', classifier, X_validate, Y_validate)
    test_pred = classifier.predict_proba(test_data)[:, 1]
    write_prediction(test_pred)
    printSHAP(classifier, X_train, [0, 5, 10])

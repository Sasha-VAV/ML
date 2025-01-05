import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", 50)

data = pd.read_csv("organisations.csv")

data = data[data.average_bill <= 2500]
data = data.loc[:, 'city':]

train_data, test_data = train_test_split(
    data, stratify=data.average_bill, test_size=0.33, random_state=42
)

train_true_data = train_data.loc[:, train_data.columns == "average_bill"]
train_data = train_data.loc[:, train_data.columns != "average_bill"]
test_true_data = test_data.loc[:, test_data.columns == "average_bill"]
test_data = test_data.loc[:, test_data.columns != "average_bill"]


def split_column(str_arr):
    str_set = set()
    for i in str_arr:
        str_set.update(i.split(" "))
    str_list = list(str_set)
    ans_dict = dict()
    for i in range(len(str_list)):
        ans_dict[str_list[i]] = i
    return ans_dict


rubrics_dict = split_column(train_data.rubrics_id.unique())
features_dict = split_column(train_data.features_id.unique())


def change_df(df: pd.DataFrame):
    df['city'] = df.city.map({'spb': 0, 'msk': 1})
    rubrics_df = pd.DataFrame({"rubric_" + str(i): 0 for i in range(len(rubrics_dict))}, index=df.index)
    rubrics_df['rubric_other'] = 0
    features_df = pd.DataFrame({"feature_" + str(i): 0 for i in range(len(features_dict))}, index=df.index)
    features_df['feature_other'] = 0
    df = pd.concat([df, rubrics_df, features_df], axis=1).copy()

    def change_row(row):
        rubrics_ids = row.rubrics_id.split(" ")
        for rubric_id in rubrics_ids:
            if rubric_id in rubrics_dict:
                row["rubric_" + str(rubrics_dict[rubric_id])] += 1
            else:
                row["rubric_other"] += 1
        features_ids = row.features_id.split(" ")
        for feature_id in features_ids:
            if feature_id in features_dict:
                row["feature_" + str(features_dict[feature_id])] += 1
            else:
                row["feature_other"] += 1
        return row

    df = df.apply(change_row, axis=1)
    df.drop(["rubrics_id", "features_id"], axis=1, inplace=True)

    return df


sparse_data_train = change_df(train_data)
sparse_data_test = change_df(test_data)

clf = CatBoostClassifier()
clf.fit(sparse_data_train, train_true_data)

clf.save_model('model', format='cbm')

y_pred = clf.predict(data=sparse_data_test)
bas = balanced_accuracy_score(test_true_data, y_pred)
print(round(bas, 2))

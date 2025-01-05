"""
source
https://new.contest.yandex.ru/contests/60376/problem?id=149944%2F2024_03_03%2FiU34rgaef0
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split

from E import MeanRegressor
from F import MostFrequentClassifier
from G import CityMeanRegressor
from H import RubricCityMedianClassifier

pd.set_option("display.max_columns", None)

data = pd.read_csv("organisations.csv")
features = pd.read_csv("features.csv")
rubrics = pd.read_csv("rubrics.csv")

features_dict = features.set_index("feature_id").to_dict()["feature_name"]
rubric_dict = rubrics.set_index("rubric_id").to_dict()["rubric_name"]

data = data[data.average_bill <= 2500]
print(f"Answer to task 3: {data.average_bill.size}")

cafe_data = data[data.rubrics_id.str.contains("30774")]

mean_msk = cafe_data[cafe_data.city == "msk"].average_bill.mean()
mean_spb = cafe_data[cafe_data.city == "spb"].average_bill.mean()
print(f"Answer to task 4: {round(mean_msk - mean_spb)}")

mean_r = data[data.rubrics_id.str.contains("30776")].average_bill.mean()
mean_p = data[data.rubrics_id.str.contains("30770")].average_bill.mean()

train_data, test_data = train_test_split(
    data, stratify=data.average_bill, test_size=0.33, random_state=42
)

# Source model is in E.py
reg = MeanRegressor()
# reg.fit(y=train_data["average_bill"])
# print(reg.predict(test_data['average_bill']))

clf = MostFrequentClassifier()


# clf.fit(y=train_data["average_bill"])
# print(clf.predict(test_data["average_bill"]))


def check_model(model, train_data, test_data, mode):
    model.fit(X=train_data.loc[:, train_data.columns != "average_bill"], y=train_data["average_bill"])
    y_pred = model.predict(test_data.loc[:, test_data.columns != "average_bill"])
    y_true = test_data["average_bill"]
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rmse = float(rmse)
    if mode == "r":
        return type(model), rmse
    bas = balanced_accuracy_score(y_true, y_pred)
    bas = float(bas)
    acs = accuracy_score(y_true, y_pred)
    return type(model), rmse, bas, acs


print(check_model(reg, train_data, test_data, "r"))
print(check_model(clf, train_data, test_data, "c"))

city_reg = CityMeanRegressor()
# city_reg.fit(X=train_data, y=train_data["average_bill"])

print(check_model(city_reg, train_data, test_data, "r"))

rubrics_id_combinations = train_data.rubrics_id.value_counts()


def filter_rubrics(row):
    rubrics_id = row["rubrics_id"]
    new_val = 'other'
    if rubrics_id in rubrics_id_combinations and rubrics_id_combinations[rubrics_id] >= 100:
        new_val = rubrics_id
    row["modified_rubrics"] = new_val
    return row


data["modified_rubrics"] = 'other'
data = data.apply(filter_rubrics, axis=1)
train_data, test_data = train_test_split(
    data, stratify=data.average_bill, test_size=0.33, random_state=42
)

r_clf = RubricCityMedianClassifier()
print(check_model(r_clf, train_data, test_data, "c"))


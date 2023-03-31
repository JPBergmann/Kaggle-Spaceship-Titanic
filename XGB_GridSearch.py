# GridSearch for LOGREG, RF, and XGB

# Imports
import os.path
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from sklearn.utils import shuffle
from xgboost import XGBClassifier

# Set "global" seed
np.random.seed(42)


def save_df(data, location):
    """
    Specify the datframe and folder inside of "Predictions" to save the predictions
    """

    version = 0
    filename = f"Predictions/{location}/v{version}.csv"

    # Check if the file already exists
    while os.path.isfile(filename):
        version += 1
        filename = f"Predictions/{location}/v{version}.csv"

    data.to_csv(filename, index=False)

    return f"Data written to file: {filename}"


train = pd.read_csv("DATA/train.csv")
test = pd.read_csv("DATA/test.csv")

X = train.drop(columns=["Transported"])
y = train["Transported"]


def create_features(df):
    df = df.copy()

    df["Cabin_Deck"] = df["Cabin"].str.split("/").str[0]
    df["Cabin_Side"] = df["Cabin"].str.split("/").str[2]

    # Create Traveling Alone Feature
    df["Group"] = df["PassengerId"].str.split("_").str[0].astype(int)
    group_sizes = df.groupby("Group").size()
    df = df.merge(group_sizes.rename("GroupSize"), left_on="Group", right_index=True)
    df["TravelingAlone"] = np.where(df["GroupSize"] > 1, False, True)
    df.drop(["Group", "GroupSize"], axis=1, inplace=True)

    return df


X = create_features(X)
test = create_features(test)

X = X.drop(columns=["Cabin", "PassengerId", "Name"])

X["CryoSleep"] = X["CryoSleep"].astype(bool)
X["VIP"] = X["VIP"].astype(bool)

test["CryoSleep"] = test["CryoSleep"].astype(bool)
test["VIP"] = test["VIP"].astype(bool)

# X = shuffle(X)
# test = shuffle(test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

transformer_num_pipe = Pipeline([("Impute", SimpleImputer(strategy="median"))])

transformer_ohe_cat_pipe = Pipeline(
    [
        ("Impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(sparse_output=False)),
    ]
)

transformer_bin_pipe = Pipeline(
    [
        ("Impute", SimpleImputer(strategy="median")),
        ("ohe", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile")),
    ]
)

transformer = ColumnTransformer(
    [
        (
            "Numeric",
            transformer_num_pipe,
            ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"],
        ),
        (
            "Categorical_OHE",
            transformer_ohe_cat_pipe,
            ["HomePlanet", "Destination", "Cabin_Deck", "Cabin_Side", "Cabin_Deck"],
        ),
        ("Binning", transformer_bin_pipe, ["Age"]),
    ]
)

# create pipe

pipe = Pipeline(
    [
        ("Preprocessing", transformer),
        ("Feature_Selection", VarianceThreshold()),
        ("Scaler", StandardScaler()),
        ("Classifier", XGBClassifier()),
    ]
)

# Initiaze the hyperparameters for each model (We exclude SVM and tree for compuatational reasons and their previous performance)

xgb_params = {}
# Pipe parameters
xgb_params["Preprocessing__Binning__ohe__n_bins"] = [5, 10]
xgb_params["Preprocessing__Numeric__Impute__strategy"] = [
    "most_frequent",
    "median",
    "mean",
]
xgb_params["Feature_Selection__threshold"] = [0, 0.25]
# Model specific parameters
xgb_params["Classifier__learning_rate"] = [0.001, 0.01]
xgb_params["Classifier__max_depth"] = [3, 6]
xgb_params["Classifier__n_estimators"] = [100, 250, 500]
xgb_params["Classifier__subsample"] = [0.5, 0.6]
xgb_params["Classifier__colsample_bytree"] = [0.5, 0.6]
xgb_params["Classifier__gamma"] = [0, 0.1, 0.2]
xgb_params["Classifier__reg_lambda"] = [0.1, 1, 10]
xgb_params["Classifier__reg_alpha"] = [0.1, 1, 10]
xgb_params["Classifier__min_child_weight"] = [1, 3, 5]

grid = GridSearchCV(
    estimator=pipe,
    param_grid=xgb_params,
    scoring="accuracy",
    n_jobs=-1,
    cv=5,
    verbose=10,
    error_score="raise",
)

grid.fit(X_train, y_train)

# Best model and params
print(f"Best params: {grid.best_params_}")

# Corresponding accuracy (cross-validated) for the above model
print(f"CV Accuracy: {grid.best_score_}")

# Testing grid-search estimator on validation set
print(f"Val Accuracy: {grid.best_estimator_.score(X_val, y_val)}")

# Predictions with best model
best_model = grid.best_estimator_
pred = best_model.predict(test.drop(columns=["Name", "PassengerId", "Cabin"])).astype(bool)
pred_results = test[["PassengerId"]]
pred_results["Transported"] = pred
save_df(data=pred_results, location="GridSearch")
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)


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


def create_features(dataframe):
    """
    Function that creates new features from the existing features in the "Spaceship Titanic" training dataset.
    Partially inspired and adapted from: https://www.kaggle.com/code/vladisov/spaceship-titanic-using-fast-ai

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe to create new features for.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with new features added.
    """

    df = dataframe.copy()

    # Numerical feature creation/transformation
    df["LogRoomService"] = np.log1p(df["RoomService"])
    df["LogFoodCourt"] = np.log1p(df["FoodCourt"])
    df["LogShoppingMall"] = np.log1p(df["ShoppingMall"])
    df["LogSpa"] = np.log1p(df["Spa"])
    df["LogVRDeck"] = np.log1p(df["VRDeck"])

    df.loc[(df["LogRoomService"].isna()) & (df["CryoSleep"] == True), "LogRoomService"] = 0.0
    df.loc[(df["LogFoodCourt"].isna()) & (df["CryoSleep"] == True), "LogFoodCourt"] = 0.0
    df.loc[(df["LogShoppingMall"].isna()) & (df["CryoSleep"] == True), "LogShoppingMall"] = 0.0
    df.loc[(df["LogSpa"].isna()) & (df["CryoSleep"] == True), "LogSpa"] = 0.0
    df.loc[(df["LogVRDeck"].isna()) & (df["CryoSleep"] == True), "LogVRDeck"] = 0.0

    df.loc[(df["RoomService"].isna()) & (df["CryoSleep"] == True), "RoomService"] = 0.0
    df.loc[(df["FoodCourt"].isna()) & (df["CryoSleep"] == True), "FoodCourt"] = 0.0
    df.loc[(df["ShoppingMall"].isna()) & (df["CryoSleep"] == True), "ShoppingMall"] = 0.0
    df.loc[(df["Spa"].isna()) & (df["CryoSleep"] == True), "Spa"] = 0.0
    df.loc[(df["VRDeck"].isna()) & (df["CryoSleep"] == True), "VRDeck"] = 0.0

    df["LogTotalSpend"] = (df["LogRoomService"] + df["LogFoodCourt"] + df["LogShoppingMall"] + df["LogSpa"] + df["LogVRDeck"])
    df["TotalSpent"] = (df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"])
    df["NoSpending"] = (df["TotalSpent"] == 0).astype(str)

    # Cabin Features
    df[["Cabin_Deck", "Cabin_Num", "Cabin_Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Cabin_Size"] = df.groupby("Cabin")["Cabin"].transform("count")

    # Age Groups (String)
    age_bins = [0, 2, 12, 18, 26, 39, 59, df["Age"].max()]
    age_labels = ["Infant", "Child", "Teenager", "Young Adult", "Adult", "Middle Aged", "Senior"]
    df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, include_lowest=True).astype(str)

    # Passenger Groups
    df["PassengerGroup"] = df["PassengerId"].str.split("_", expand=True)[0]
    df["GroupSize"] = df.groupby("PassengerGroup")["PassengerGroup"].transform("count")

    # NA's
    df.loc[(df["CryoSleep"].isna()) & (df["TotalSpent"] > 0), "CryoSleep"] = False  # Cant spend money if you are in cryosleep
    df.loc[(df["CryoSleep"].isna()) & (df["TotalSpent"] == 0), "CryoSleep"] = True  # If you haven't spent money you are very likely in cryosleep

    df.loc[(df["HomePlanet"].isna()) & (df["Cabin_Deck"] == "G"), "HomePlanet"] = "Earth" # All G deck passengers are from Earth (see EDA)
    df.loc[(df["HomePlanet"].isna()) & (df["Cabin_Deck"].isin(["B", "A", "C", "T"])), "HomePlanet"] = "Europa" # All B, A, C, T deck passengers are from Europa (see EDA)

    df = df.drop(columns=["Cabin", "PassengerId", "Name", "Age", "PassengerGroup", "Cabin_Num"])

    return df


class IQRTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer that handles outliers from numerical features using the IQR method.

    Parameters
    ----------
    handle : str, optional
        How to handle the outliers:
        If None, no action is taken. 
        If "bound", the outliers are bounded to the upper and lower bound.
        If "replace", the outliers are replaced with NaNs (to be handled by the SimpleImputer).
    """

    def __init__(self, handle=None):
        self.handle = handle

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num_cols = X.select_dtypes(include=np.number).columns

        if not self.handle:
            return X
        
        for col in num_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            S = 1.5 * IQR
            LB = Q1 - S
            UB = Q3 + S

            if self.handle == "replace":
                X.loc[X[col] > UB, col] = np.nan
                X.loc[X[col] < LB, col] = np.nan

            elif self.handle == "bound":
                X.loc[X[col] > UB, col] = UB
                X.loc[X[col] < LB, col] = LB

        return X


train = pd.read_csv("DATA/train.csv")
test = pd.read_csv("DATA/test.csv")
X = train.drop(columns=["Transported"])
y = train["Transported"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, X_test = create_features(X_train), create_features(X_val), create_features(test)

cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()


transformer_num_pipe = Pipeline(
    [("Impute", SimpleImputer(strategy="median"))]
)

transformer_ohe_cat_pipe = Pipeline(
    [
        ("Impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(sparse_output=False)),
    ]
)

transformer = ColumnTransformer(
    [
        ("Numeric", transformer_num_pipe, num_cols),
        ("Categorical_OHE", transformer_ohe_cat_pipe, cat_cols),
    ],
    remainder="passthrough",
)


CAT = CatBoostClassifier(random_state=SEED, verbose=False)
XGB = XGBClassifier(random_state=SEED)
RF = RandomForestClassifier(random_state=SEED)


pipe = Pipeline(
    [
        ("IQR", IQRTransformer()),
        ("Preprocessing", transformer),
        ("Feature_Selection", VarianceThreshold()),
        ("Scaler", MinMaxScaler()),
        ("Clf", CAT)
    ])

cat_params = {}
# Pipe Params
cat_params["IQR__handle"] = [None, "replace", "bound"]
cat_params["Preprocessing__Numeric__Impute__strategy"] = ["mean", "median", "most_frequent"]
cat_params["Feature_Selection__threshold"] = [0, 0.1, 0.25]
cat_params["Scaler__feature_range"] = [(0, 1), (-1, 1)]
# Model Params
cat_params["Clf__learning_rate"] = [0.1, 0.01, 0.001]
cat_params["Clf__iterations"] = [100, 500, 1000]
cat_params["Clf__depth"] = [4, 6, 8]
cat_params["Clf__l2_leaf_reg"] = [1, 3]
cat_params["Clf__subsample"] = [0.8, 0.9, 1.0]
cat_params["Clf__colsample_bylevel"] = [0.8, 0.9, 1.0]
cat_params["Clf"] = [CAT]

xgb_params = {}
# Pipe Params
xgb_params["IQR__handle"] = [None, "replace", "bound"]
xgb_params["Preprocessing__Numeric__Impute__strategy"] = ["mean", "median", "most_frequent"]
xgb_params["Feature_Selection__threshold"] = [0, 0.1, 0.25]
xgb_params["Scaler__feature_range"] = [(0, 1), (-1, 1)]
# Model Params
xgb_params["Clf__learning_rate"] = [0.01, 0.1]
xgb_params["Clf__max_depth"] = [3, 6, 9]
xgb_params["Clf__n_estimators"] = [100, 500]
xgb_params["Clf__subsample"] = [0.8, 0.9, 1.0]
xgb_params["Clf__colsample_bytree"] = [0.8, 0.9, 1.0]
xgb_params["Clf__gamma"] = [0, 0.1]
xgb_params["Clf"] = [XGB]

rf_params = {}
# Pipe Params
rf_params["IQR__handle"] = [None, "replace", "bound"]
rf_params["Preprocessing__Numeric__Impute__strategy"] = ["mean", "median", "most_frequent"]
rf_params["Feature_Selection__threshold"] = [0, 0.1, 0.25]
rf_params["Scaler__feature_range"] = [(0, 1), (-1, 1)]
# Model Params
rf_params["Clf__n_estimators"] = [100, 300]
rf_params["Clf__max_depth"] = [10, 30, 50]
rf_params["Clf__min_samples_split"] = [2, 5]
rf_params["Clf__min_samples_leaf"] = [1, 2]
rf_params["Clf__max_features"] = ["sqrt", "log2"]
rf_params["Clf"] = [RF]


params = [cat_params, xgb_params, rf_params]

grid = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    scoring="accuracy",
    n_jobs=-2,
    cv=3,
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
test_preds = best_model.predict(X_test).astype(bool)
preds = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": test_preds})

save_df(data=preds, location="MultimodelGrid")

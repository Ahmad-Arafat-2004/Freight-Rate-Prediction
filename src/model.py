import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

from src.config import SEED
from src.features import CATEGORICAL, TIME_FEATURES

LGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
)


class HybridModel:
    """log(rate) = linear(time features) + LightGBM(static features), fitted by backfitting."""

    def __init__(self, static_features, n_iter=3, lgb_params=None):
        self.static_features = list(static_features)
        self.n_iter = n_iter
        self.lgb_params = lgb_params or LGB_PARAMS
        self.categories = {}

    def _static_matrix(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = df[self.static_features].copy()
        for col in [c for c in CATEGORICAL if c in X.columns]:
            if fit:
                self.categories[col] = sorted(X[col].dropna().unique())
            known = X[col].where(X[col].isin(self.categories[col]))  # unseen -> NaN
            X[col] = pd.Categorical(known, categories=self.categories[col])
        return X

    def fit(self, df: pd.DataFrame, y: pd.Series):
        X_static = self._static_matrix(df, fit=True)
        X_time = df[TIME_FEATURES]
        y = np.asarray(y)

        self.linear = LinearRegression()
        linear_part = np.zeros(len(df))
        for _ in range(self.n_iter):
            self.gbm = lgb.LGBMRegressor(**self.lgb_params).fit(X_static, y - linear_part)
            gbm_part = self.gbm.predict(X_static)
            self.linear.fit(X_time, y - gbm_part)
            linear_part = self.linear.predict(X_time)
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        X_static = self._static_matrix(df)
        return self.gbm.predict(X_static) + self.linear.predict(df[TIME_FEATURES])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))

    def time_coefficients(self) -> dict:
        return dict(zip(TIME_FEATURES, self.linear.coef_))



class CityAwareModel:
    """Uses city identity when both cities were seen in training, otherwise falls back to a model without it."""

    def __init__(self, static_features, city_features, n_iter=3, lgb_params=None):
        self.with_city = HybridModel(list(static_features) + list(city_features), n_iter, lgb_params)
        self.without_city = HybridModel(static_features, n_iter, lgb_params)

    def fit(self, df, y):
        self.known_cities = set(df["pickup"]) | set(df["delivery"])
        self.with_city.fit(df, y)
        self.without_city.fit(df, y)
        return self

    def uses_city_model(self, df):
        return (df["pickup"].isin(self.known_cities) & df["delivery"].isin(self.known_cities)).to_numpy()

    def predict(self, df):
        known = self.uses_city_model(df)
        return np.where(known, self.with_city.predict(df), self.without_city.predict(df))

    def time_coefficients(self):
        return self.without_city.time_coefficients()
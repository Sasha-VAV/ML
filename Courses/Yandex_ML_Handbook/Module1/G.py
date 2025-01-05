from sklearn.base import RegressorMixin


class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        cities = X.city.unique()
        X['target'] = y
        self.mean_city = dict()
        for city in cities:
            city_series = X[X.city == city].target
            mean_city = city_series.mean()
            self.mean_city[city] = mean_city

    def predict(self, X=None):
        return X.city.apply(lambda x: self.mean_city[x])

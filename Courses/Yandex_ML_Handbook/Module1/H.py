from sklearn.base import ClassifierMixin


class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        df = X
        df['average_bill'] = y if y is not None else df['average_bill']
        self.med = df.groupby(['city', 'modified_rubrics'])['average_bill'].median()

    def predict(self, X=None):
        return X.apply(lambda x: self.med.get((x.city, x.modified_rubrics)), axis=1)

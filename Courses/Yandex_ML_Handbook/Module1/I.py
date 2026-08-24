from sklearn.base import ClassifierMixin


class LargeClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        df = X
        df['average_bill'] = y if y is not None else df['average_bill']
        mfs = df.modified_features.unique()
        self.global_med = df.average_bill.median()
        self.med_dict = dict()
        for mf in mfs:
            self.med_dict[mf] = df[df.modified_features == mf].average_bill.median()

    def predict(self, X=None):
        def select_mf(x):
            if x in self.med_dict:
                return self.med_dict[x]
            else:
                return self.global_med

        return X.apply(lambda k: select_mf(k.modified_features), axis=1)

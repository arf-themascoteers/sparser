import pandas as pd
from sklearn.model_selection import train_test_split
from data_splits import DataSplits
import numpy as np


class DSManager:
    def __init__(self, name, folds=1):
        self.name = name
        self.folds = folds
        self.init_seed = 40
        self.random_state_start = 80
        self._reset_seed()
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        df.iloc[:, 0], class_labels = pd.factorize(df.iloc[:, 0])
        self.full_data = df.to_numpy()
        #train:validation:evaluation_train:evaluation_test = 0.45:  0.0.5:  0.50    :0.50

    def get_name(self):
        return self.name

    def count_rows(self):
        return self.full_data.shape[0]

    def count_features(self):
        return self.full_data.shape[1]-1
    
    def _shuffle(self, seed):
        self._set_seed(seed)
        shuffled_indices = np.random.permutation(self.full_data.shape[0])
        self._reset_seed()
        return self.full_data[shuffled_indices]

    def get_k_folds(self):
        for i in range(self.folds):
            seed = self.random_state_start + i
            yield self.get_all_set_X_y_from_data(seed)

    def get_all_set_X_y_from_data(self, seed):
        data = self._shuffle(seed)
        train_validation, evaluation = train_test_split(data, test_size=0.5, random_state=seed)
        train, validation = train_test_split(train_validation, test_size=0.1, random_state=seed)
        evaluation_train, evaluation_test = train_test_split(train_validation, test_size=0.5, random_state=seed)
        return DataSplits(self.name, *DSManager.get_X_y_from_data(train),
                          *DSManager.get_X_y_from_data(validation),
                          *DSManager.get_X_y_from_data(evaluation_train),
                          *DSManager.get_X_y_from_data(evaluation_test)
                          )

    def _set_seed(self, seed):
        np.random.seed(seed)

    def _reset_seed(self):
        np.random.seed(self.init_seed)

    def __repr__(self):
        return self.get_name()

    @staticmethod
    def get_X_y_from_data(data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    @staticmethod
    def get_dataset_names():
        return [
            "indian_pines",
            "paviaU",
            "salinasA",

            "pavia",
            "salinas"
        ]


if __name__ == "__main__":
    ds = DSManager("indian_pines")
    for split in ds.get_k_folds():
        print(split.splits_description())

    ds = DSManager("indian_pines",10)
    for split in ds.get_k_folds():
        print(split.splits_description())
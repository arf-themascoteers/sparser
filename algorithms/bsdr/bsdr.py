import torch
from algorithms.bsdr.ann import ANN
from datetime import datetime
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import numpy as np
from algorithms.bsdr.linterp import LinearInterpolationModule
import train_test_evaluator


class BSDR:
    def __init__(self, target_size, class_size, split, machine_name, repeat, fold, structure=None, verbose=True, epochs=2000):
        self.target_size = target_size
        self.class_size = class_size
        self.split = split
        self.machine_name = machine_name
        self.repeat = repeat
        self.fold = fold
        self.verbose = verbose
        self.lr = 0.001
        self.model = ANN(self.target_size, self.class_size, structure)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = self.get_criterion()
        self.epochs = epochs
        self.csv_file = os.path.join("results", f"{self.machine_name}-{self.split.get_name()}-{target_size}-{self.repeat}-{self.fold}.csv")
        self.original_feature_size = None
        self.start_time = datetime.now()
        print(f"MACHINE: {self.machine_name}")

    def get_criterion(self):
        if self.is_regression():
            return torch.nn.MSELoss(reduction='mean')
        return torch.nn.CrossEntropyLoss()

    def is_regression(self):
        return self.class_size == 1

    def get_elapsed_time(self):
        elapsed_time = round((datetime.now() - self.start_time).total_seconds(),2)
        return float(elapsed_time)

    def create_optimizer(self):
        weight_decay = self.lr/10
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        return optimizer, scheduler

    def fit(self, X, y, X_validation, y_validation):
        self.original_feature_size = X.shape[1]
        if self.verbose:
            self.write_columns()
        self.model.train()
        optimizer, scheduler = self.create_optimizer()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        linterp = LinearInterpolationModule(X, self.device)
        X_validation = torch.tensor(X_validation, dtype=torch.float32).to(self.device)
        linterp_validation = LinearInterpolationModule(X_validation, self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        y_validation = torch.tensor(y_validation, dtype=torch.float32).to(self.device)
        y = y.type(torch.LongTensor).to(self.device)
        y_validation = y_validation.type(torch.LongTensor).to(self.device)
        for epoch in range(self.epochs):
            y_hat = self.model(linterp)
            loss = self.criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #scheduler.step()

            if self.verbose:
                row = self.dump_row(epoch, optimizer, linterp, y, linterp_validation, y_validation)
                row = [round(item, 5) if isinstance(item, float) else item for item in row]
                if epoch%50 == 0:
                    print("".join([str(i).ljust(20) for i in row]))
        return self.get_indices()

    def evaluate(self,spline,y):
        self.model.eval()
        y_hat = self.model(spline)
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y_hat = np.argmax(y_hat, axis=1)
        accuracy = accuracy_score(y, y_hat)
        aa = train_test_evaluator.average_accuracy(y, y_hat)
        kappa = cohen_kappa_score(y, y_hat)
        self.model.train()
        return accuracy, aa, kappa

    def write_columns(self):
        columns = "epoch,train_oa,validation_oa,train_aa,validation_aa,train_k,validation_k,time,lr".split(",")
        for index,p in enumerate(self.model.get_indices()):
            columns.append(f"band_{index+1}")
        print("".join([c.ljust(20) for c in columns]))
        with open(self.csv_file, 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    def get_current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def dump_row(self, epoch, optimizer, spline, y, spline_validation, y_validation):
        current_lr = self.get_current_lr(optimizer)
        train_oa, train_aa, train_k = self.evaluate(spline, y)
        test_oa, test_aa, test_k = self.evaluate(spline_validation, y_validation)
        row = [train_oa, test_oa, train_aa, test_aa, train_k, test_k]
        row = [r for r in row]
        elapsed_time = self.get_elapsed_time()
        row = [epoch] + row + [elapsed_time, current_lr] + self.get_indices()
        with open(self.csv_file, 'a') as file:
            file.write(",".join([f"{x}" for x in row]))
            file.write("\n")
        return row

    def get_indices(self):
        indices = torch.round(self.model.get_indices() * self.original_feature_size ).to(torch.int64).tolist()
        return list(dict.fromkeys(indices))

    def transform(self, X):
        return X[:,self.get_indices()]

import numpy as np


class AdalineGD:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]+1)
        self.cost_ = []
        for _ in range(self.n_iter):
            errors = y - self.activation(self.net_input(X))
            updates = self.eta * X.T.dot(errors)
            self.w_[1:] += updates
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append((errors ** 2).sum() / 2)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

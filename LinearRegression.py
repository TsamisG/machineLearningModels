import numpy as np

class LinearRegressor:
    def __init__(self, input_dim, output_dim, alpha=1e-04, epochs=400, bias=True):
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.has_bias = bias
        self.b = np.zeros(output_dim)
        self.alpha = alpha
        self.epochs = epochs

    def predict(self, x):
        return x.dot(self.W) + self.b

    def gradW(self, y, yhat, x):
        return x.T.dot(yhat - y)

    def gradb(self, y, yhat):
        return (yhat - y).sum(axis=0)

    def partial_fit(self, x, y):
        yhat = self.predict(x)
        self.W -= self.alpha * self.gradW(y, yhat, x)
        self.b -= self.has_bias * self.alpha * self.gradb(y, yhat)
    
    def fit(self, x, y):
        costs = []
        for ep in range(self.epochs):
            yhat = self.predict(x)
            costs.append(self.cost(x, y))
            self.W -= self.alpha * self.gradW(y, yhat, x)
            self.b -= self.has_bias * self.alpha * self.gradb(y, yhat)
        return costs

    def cost(self, x, y):
        yhat = self.predict(x)
        return np.mean((y - yhat)**2)



class LogisticRegressor:
    def __init__(self, input_dim, alpha=1e-04, epochs=400, bias=True):
        self.W = np.random.randn(input_dim,) / np.sqrt(input_dim)
        self.has_bias = bias
        self.b = 0
        self.alpha = alpha
        self.epochs = epochs

    def forward(self, x):
        a = x.dot(self.W) + self.b
        return 1./(1. + np.exp(-a))

    def predict(self, x):
        p_y = self.forward(x)
        return np.argmax(p_y, axis=0)

    def gradW(self, y, yhat, x):
        return x.T.dot(yhat - y)

    def gradb(self, y, yhat):
        return (yhat - y).sum(axis=0)

    def partial_fit(self, x, y):
        p_y = self.forward(x)
        self.W -= self.alpha * self.gradW(y, p_y, x)
        self.b -= self.has_bias * self.alpha * self.gradb(y, p_y)
    
    def fit(self, x, y):
        costs = []
        for ep in range(self.epochs):
            p_y = self.forward(x)
            costs.append(self.cross_entropy(y, p_y))
            self.W -= self.alpha * self.gradW(y, p_y, x)
            self.b -= self.has_bias * self.alpha * self.gradb(y, p_y)
        return costs

    def cross_entropy(self, y, p_y):
        N = len(y)
        E = 0
        for n in range(N):
            if y[n] == 0:
                E -= (1-y[n])*np.log(1-p_y[n])
            else:
                E -= y[n]*np.log(p_y[n])
        return E

    def error_rate(self, x, y):
        yhat = self.predict(x)
        return np.mean(yhat != y)
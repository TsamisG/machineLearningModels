import numpy as np
from sklearn.utils import shuffle


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class Layer:
    def __init__(self, M1, M2, act_fun, islast=False):
        self.islast = islast
        self.act_fun = act_fun
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = W
        self.b = b
        self.cacheW = 0
        self.cacheb = 0
        self.mW = 0
        self.mb = 0
        self.vW = 0
        self.vb = 0

    def forward(self, X):
        a = X.dot(self.W) + self.b
        if not self.islast:
            if self.act_fun == 'relu' :
                a[a < 0] = 0
            elif self.act_fun =='sigmoid':
                a = sigmoid(a)
        return a


    def grad(self, delta, Z_prev, Z=None, W_next=0):
        if self.islast:
            delta = delta
        else:
            if self.act_fun == 'relu' :
                dZ = Z > 0
            elif self.act_fun =='sigmoid':
                dZ = sigmoid(Z) * (1 - sigmoid(Z))
            delta = (delta.dot(W_next.T)) * dZ
        return Z_prev.T.dot(delta), delta.sum(axis=0), delta



class ANNClassification:
    def __init__(self, input_size, output_size, hidden_layer_sizes, learning_rate=1e-04, reg=0.1,
             epochs=400, batch_size=10, act_fun='relu', verbose=True):
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.act_fun = act_fun
        self.verbose = verbose
        self.layers = []
        K = output_size
        M1 = input_size
        for M2 in self.hidden_layer_sizes:
            h = Layer(M1, M2, self.act_fun)
            self.layers.append(h)
            M1 = M2
        h = Layer(M1, K, self.act_fun, islast=True)
        self.layers.append(h)
        self.t = 0

    
    def partial_fit(self, X, Y):
        Yhat, Z = self.forward(X)
        
        delta = Yhat-Y
        gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
        
        self.layers[-1].W -= self.learning_rate * (gradW + self.reg*self.layers[-1].W)
        self.layers[-1].b -= self.learning_rate * gradb
        
        for i in range(len(self.layers)-2,-1,-1):
            gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
            self.layers[i].W -= self.learning_rate * (gradW + self.reg*self.layers[i].W)
            self.layers[i].b -= self.learning_rate * gradb


    def partial_fit_RMSProp(self, X, Y, decay=0.99, mu=0.9):
        eps = 1e-08
        Yhat, Z = self.forward(X)
        delta = Yhat-Y
        gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
        self.layers[-1].cacheW = decay * self.layers[-1].cacheW + (1-decay) * gradW * gradW
        self.layers[-1].cacheb = decay * self.layers[-1].cacheb + (1-decay) * gradb * gradb
        self.layers[-1].vW = mu * self.layers[-1].vW + (1-mu) * self.learning_rate * (gradW + self.reg*self.layers[-1].W) / (np.sqrt(self.layers[-1].cacheW) + eps)
        self.layers[-1].vb = mu * self.layers[-1].vb + (1-mu) * self.learning_rate * gradb / (np.sqrt(self.layers[-1].cacheb) + eps)
        self.layers[-1].W -= self.layers[-1].vW
        self.layers[-1].b -= self.layers[-1].vb
        
        for i in range(len(self.layers)-2,-1,-1):
            gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
            self.layers[i].cacheW = decay * self.layers[i].cacheW + (1-decay) * gradW * gradW
            self.layers[i].cacheb = decay * self.layers[i].cacheb + (1-decay) * gradb * gradb
            self.layers[i].vW = mu * self.layers[i].vW + (1-mu) * self.learning_rate * (gradW + self.reg*self.layers[i].W) / (np.sqrt(self.layers[i].cacheW) + eps)
            self.layers[i].vb = mu * self.layers[i].vb + (1-mu) * self.learning_rate * gradb / (np.sqrt(self.layers[i].cacheb) + eps)
            self.layers[i].W -= self.layers[i].vW
            self.layers[i].b -= self.layers[i].vb

    
    def partial_fit_Adam(self, X, Y, beta1=0.9, beta2=0.999):
        eps = 1e-08
        self.t += 1
        Yhat, Z = self.forward(X)

        delta = Yhat-Y
        gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
        self.layers[-1].mW = beta1 * self.layers[-1].mW + (1-beta1) * gradW
        self.layers[-1].mb = beta1 * self.layers[-1].mb + (1-beta1) * gradb
        self.layers[-1].vW = beta2 * self.layers[-1].vW + (1-beta2) * gradW * gradW
        self.layers[-1].vb = beta2 * self.layers[-1].vb + (1-beta2) * gradb * gradb
        mW_hat = self.layers[-1].mW / (1-beta1**self.t)
        mb_hat = self.layers[-1].mb / (1-beta1**self.t)
        vW_hat = self.layers[-1].vW / (1-beta2**self.t)
        vb_hat = self.layers[-1].vb / (1-beta2**self.t)
        self.layers[-1].W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
        self.layers[-1].b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
        
        for i in range(len(self.layers)-2,-1,-1):
            gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
            self.layers[i].mW = beta1 * self.layers[i].mW + (1-beta1) * gradW
            self.layers[i].mb = beta1 * self.layers[i].mb + (1-beta1) * gradb
            self.layers[i].vW = beta2 * self.layers[i].vW + (1-beta2) * gradW * gradW
            self.layers[i].vb = beta2 * self.layers[i].vb + (1-beta2) * gradb * gradb
            mW_hat = self.layers[i].mW / (1-beta1**self.t)
            mb_hat = self.layers[i].mb / (1-beta1**self.t)
            vW_hat = self.layers[i].vW / (1-beta2**self.t)
            vb_hat = self.layers[i].vb / (1-beta2**self.t)
            self.layers[i].W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
            self.layers[i].b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

    
    
    def fit_GD(self, X, Y, Xval, Yval):
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Xval = Xval.astype(np.float32)
        
        N, D = X.shape
        n_batches = int(np.ceil(N / self.batch_size))
        
        self.train_losses = []
        self.val_losses = []
        
        for ep in range(1, self.epochs+1):
            for j in range(n_batches):
                Xbatch = X[j*n_batches:(j+1)*n_batches,:]
                Ybatch = Y[j*n_batches:(j+1)*n_batches,:]
                
                pYbatch, Z = self.forward(Xbatch)

                delta = pYbatch-Ybatch
                gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
                
                self.layers[-1].W -= self.learning_rate * (gradW + self.reg*self.layers[-1].W)
                self.layers[-1].b -= self.learning_rate * gradb
                
                for i in range(len(self.layers)-2,-1,-1):
                    gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
                    self.layers[i].W -= self.learning_rate * (gradW + self.reg*self.layers[i].W)
                    self.layers[i].b -= self.learning_rate * gradb

            pYtrain, _ = self.forward(X)
            train_loss = self.loss(Y, pYtrain)
            self.train_losses.append(train_loss)
            
            pYval, _ = self.forward(Xval)
            val_loss = self.loss(Yval, pYval)
            self.val_losses.append(val_loss)
            
            if self.verbose and (ep == 1 or ep % 20 == 0):
                c_rate = self.classification_rate(Xval, Yval)
                print(f'Epoch: [ {ep} / {self.epochs} ] | Training Loss: {train_loss:.2f} | Val. Loss: {val_loss:.2f} | Val. Classification Rate: {c_rate*100:.2f}%')
    
    
    def fit_RMSProp(self, X, Y, Xval, Yval, decay=0.99, mu=0.9):
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Xval = Xval.astype(np.float32)
        
        N, D = X.shape
        n_batches = int(np.ceil(N / self.batch_size))
        
        self.train_losses = []
        self.val_losses = []
        
        eps = 1e-08
        
        for ep in range(1, self.epochs+1):
            for j in range(n_batches):
                Xbatch = X[j*n_batches:(j+1)*n_batches,:]
                Ybatch = Y[j*n_batches:(j+1)*n_batches,:]
                
                pYbatch, Z = self.forward(Xbatch)

                delta = pYbatch-Ybatch
                gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
                self.layers[-1].cacheW = decay * self.layers[-1].cacheW + (1-decay) * gradW * gradW
                self.layers[-1].cacheb = decay * self.layers[-1].cacheb + (1-decay) * gradb * gradb
                self.layers[-1].vW = mu * self.layers[-1].vW + (1-mu) * self.learning_rate * (gradW + self.reg*self.layers[-1].W) / (np.sqrt(self.layers[-1].cacheW) + eps)
                self.layers[-1].vb = mu * self.layers[-1].vb + (1-mu) * self.learning_rate * gradb / (np.sqrt(self.layers[-1].cacheb) + eps)
                self.layers[-1].W -= self.layers[-1].vW
                self.layers[-1].b -= self.layers[-1].vb
                
                for i in range(len(self.layers)-2,-1,-1):
                    gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
                    self.layers[i].cacheW = decay * self.layers[i].cacheW + (1-decay) * gradW * gradW
                    self.layers[i].cacheb = decay * self.layers[i].cacheb + (1-decay) * gradb * gradb
                    self.layers[i].vW = mu * self.layers[i].vW + (1-mu) * self.learning_rate * (gradW + self.reg*self.layers[i].W) / (np.sqrt(self.layers[i].cacheW) + eps)
                    self.layers[i].vb = mu * self.layers[i].vb + (1-mu) * self.learning_rate * gradb / (np.sqrt(self.layers[i].cacheb) + eps)
                    self.layers[i].W -= self.layers[i].vW
                    self.layers[i].b -= self.layers[i].vb

            pYtrain, _ = self.forward(X)
            train_loss = self.loss(Y, pYtrain)
            self.train_losses.append(train_loss)
            
            pYval, _ = self.forward(Xval)
            val_loss = self.loss(Yval, pYval)
            self.val_losses.append(val_loss)
            
            if self.verbose and (ep == 1 or ep % 20 == 0):
                c_rate = self.classification_rate(Xval, Yval)
                print(f'Epoch: [ {ep} / {self.epochs} ] | Training Loss: {train_loss:.2f} | Val. Loss: {val_loss:.2f} | Val. Classification Rate: {c_rate*100:.2f}%')
    

    def fit_Adam(self, X, Y, Xval, Yval, beta1=0.9, beta2=0.999):
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Xval = Xval.astype(np.float32)
        
        N, D = X.shape
        n_batches = int(np.ceil(N / self.batch_size))
        
        self.train_losses = []
        self.val_losses = []
        
        eps = 1e-08
        for ep in range(1, self.epochs+1):
            for j in range(n_batches):
                Xbatch = X[j*n_batches:(j+1)*n_batches,:]
                Ybatch = Y[j*n_batches:(j+1)*n_batches,:]
                self.t += 1
                pYbatch, Z = self.forward(Xbatch)

                delta = pYbatch-Ybatch
                gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
                self.layers[-1].mW = beta1 * self.layers[-1].mW + (1-beta1) * gradW
                self.layers[-1].mb = beta1 * self.layers[-1].mb + (1-beta1) * gradb
                self.layers[-1].vW = beta2 * self.layers[-1].vW + (1-beta2) * gradW * gradW
                self.layers[-1].vb = beta2 * self.layers[-1].vb + (1-beta2) * gradb * gradb
                mW_hat = self.layers[-1].mW / (1-beta1**self.t)
                mb_hat = self.layers[-1].mb / (1-beta1**self.t)
                vW_hat = self.layers[-1].vW / (1-beta2**self.t)
                vb_hat = self.layers[-1].vb / (1-beta2**self.t)
                self.layers[-1].W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
                self.layers[-1].b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
                
                for i in range(len(self.layers)-2,-1,-1):
                    gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
                    self.layers[i].mW = beta1 * self.layers[i].mW + (1-beta1) * gradW
                    self.layers[i].mb = beta1 * self.layers[i].mb + (1-beta1) * gradb
                    self.layers[i].vW = beta2 * self.layers[i].vW + (1-beta2) * gradW * gradW
                    self.layers[i].vb = beta2 * self.layers[i].vb + (1-beta2) * gradb * gradb
                    mW_hat = self.layers[i].mW / (1-beta1**self.t)
                    mb_hat = self.layers[i].mb / (1-beta1**self.t)
                    vW_hat = self.layers[i].vW / (1-beta2**self.t)
                    vb_hat = self.layers[i].vb / (1-beta2**self.t)
                    self.layers[i].W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
                    self.layers[i].b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
                
                
            
            pYtrain, _ = self.forward(X)
            train_loss = self.loss(Y, pYtrain)
            self.train_losses.append(train_loss)
            
            pYval, _ = self.forward(Xval)
            val_loss = self.loss(Yval, pYval)
            self.val_losses.append(val_loss)
            
            if self.verbose and (ep == 1 or ep % 20 == 0):
                c_rate = self.classification_rate(Xval, Yval)
                print(f'Epoch: [ {ep} / {self.epochs} ] | Training Loss: {train_loss:.2f} | Val. Loss: {val_loss:.2f} | Val. Classification Rate: {c_rate*100:.2f}%')

    
    def forward(self, X):
        Z = [X]
        for h in self.layers[:-1]:
            Z.append(h.forward(Z[-1]))
        
        a = self.layers[-1].forward(Z[-1])
        pY = np.exp(a)
        pY = pY / np.sum(pY, axis=1, keepdims=True)
        return pY, Z

    def loss(self, Y, pY):
        loss = Y*np.log(pY)
        return -loss.sum()

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.round(pY)

    def classification_rate(self, X, Y):
        pred = np.argmax(self.predict(X), axis=1)
        Y = np.argmax(Y, axis=1)
        return np.mean(pred == Y)



class ANNRegression:
    def __init__(self, input_size, output_size, hidden_layer_sizes, learning_rate=1e-04, reg=0.1,
                 epochs=400, batch_size=10, act_fun='relu', verbose=True):
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.act_fun = act_fun
        self.verbose = verbose
        self.layers = []
        M = output_size
        M1 = input_size
        for M2 in self.hidden_layer_sizes:
            h = Layer(M1, M2, self.act_fun)
            self.layers.append(h)
            M1 = M2
        h = Layer(M1, M, self.act_fun, islast=True)
        self.layers.append(h)
        self.t = 0

    def partial_fit(self, X, Y):
        Yhat, Z = self.forward(X)
        
        delta = Yhat-Y
        gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
        
        self.layers[-1].W -= self.learning_rate * (gradW + self.reg*self.layers[-1].W)
        self.layers[-1].b -= self.learning_rate * gradb
        
        for i in range(len(self.layers)-2,-1,-1):
            gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
            self.layers[i].W -= self.learning_rate * (gradW + self.reg*self.layers[i].W)
            self.layers[i].b -= self.learning_rate * gradb


    def partial_fit_RMSProp(self, X, Y, decay=0.99, mu=0.9):
        eps = 1e-08
        Yhat, Z = self.forward(X)
        delta = Yhat-Y
        gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
        self.layers[-1].cacheW = decay * self.layers[-1].cacheW + (1-decay) * gradW * gradW
        self.layers[-1].cacheb = decay * self.layers[-1].cacheb + (1-decay) * gradb * gradb
        self.layers[-1].vW = mu * self.layers[-1].vW + (1-mu) * self.learning_rate * (gradW + self.reg*self.layers[-1].W) / (np.sqrt(self.layers[-1].cacheW) + eps)
        self.layers[-1].vb = mu * self.layers[-1].vb + (1-mu) * self.learning_rate * gradb / (np.sqrt(self.layers[-1].cacheb) + eps)
        self.layers[-1].W -= self.layers[-1].vW
        self.layers[-1].b -= self.layers[-1].vb
        
        for i in range(len(self.layers)-2,-1,-1):
            gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
            self.layers[i].cacheW = decay * self.layers[i].cacheW + (1-decay) * gradW * gradW
            self.layers[i].cacheb = decay * self.layers[i].cacheb + (1-decay) * gradb * gradb
            self.layers[i].vW = mu * self.layers[i].vW + (1-mu) * self.learning_rate * (gradW + self.reg*self.layers[i].W) / (np.sqrt(self.layers[i].cacheW) + eps)
            self.layers[i].vb = mu * self.layers[i].vb + (1-mu) * self.learning_rate * gradb / (np.sqrt(self.layers[i].cacheb) + eps)
            self.layers[i].W -= self.layers[i].vW
            self.layers[i].b -= self.layers[i].vb

    
    def partial_fit_Adam(self, X, Y, beta1=0.9, beta2=0.999):
        eps = 1e-08
        self.t += 1
        Yhat, Z = self.forward(X)

        delta = Yhat-Y
        gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
        self.layers[-1].mW = beta1 * self.layers[-1].mW + (1-beta1) * gradW
        self.layers[-1].mb = beta1 * self.layers[-1].mb + (1-beta1) * gradb
        self.layers[-1].vW = beta2 * self.layers[-1].vW + (1-beta2) * gradW * gradW
        self.layers[-1].vb = beta2 * self.layers[-1].vb + (1-beta2) * gradb * gradb
        mW_hat = self.layers[-1].mW / (1-beta1**self.t)
        mb_hat = self.layers[-1].mb / (1-beta1**self.t)
        vW_hat = self.layers[-1].vW / (1-beta2**self.t)
        vb_hat = self.layers[-1].vb / (1-beta2**self.t)
        self.layers[-1].W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
        self.layers[-1].b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
        
        for i in range(len(self.layers)-2,-1,-1):
            gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
            self.layers[i].mW = beta1 * self.layers[i].mW + (1-beta1) * gradW
            self.layers[i].mb = beta1 * self.layers[i].mb + (1-beta1) * gradb
            self.layers[i].vW = beta2 * self.layers[i].vW + (1-beta2) * gradW * gradW
            self.layers[i].vb = beta2 * self.layers[i].vb + (1-beta2) * gradb * gradb
            mW_hat = self.layers[i].mW / (1-beta1**self.t)
            mb_hat = self.layers[i].mb / (1-beta1**self.t)
            vW_hat = self.layers[i].vW / (1-beta2**self.t)
            vb_hat = self.layers[i].vb / (1-beta2**self.t)
            self.layers[i].W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
            self.layers[i].b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
    
    
    def fit_GD(self, X, Y, Xval, Yval):
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        Xval = Xval.astype(np.float32)
        Yval = Yval.astype(np.float32)
        
        N, D = X.shape
        n_batches = int(np.ceil(N / self.batch_size))
        
        self.train_losses = []
        self.val_losses = []
        
        for ep in range(1, self.epochs+1):
            for j in range(n_batches):
                Xbatch = X[j*n_batches:(j+1)*n_batches,:]
                Ybatch = Y[j*n_batches:(j+1)*n_batches,:]
                
                Yhat_batch, Z = self.forward(Xbatch)

                delta = Yhat_batch-Ybatch
                gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
                
                self.layers[-1].W -= self.learning_rate * (gradW + self.reg*self.layers[-1].W)
                self.layers[-1].b -= self.learning_rate * gradb
                
                for i in range(len(self.layers)-2,-1,-1):
                    gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
                    self.layers[i].W -= self.learning_rate * (gradW + self.reg*self.layers[i].W)
                    self.layers[i].b -= self.learning_rate * gradb

            Yhat_train, _ = self.forward(X)
            train_loss = self.loss(Y, Yhat_train)
            self.train_losses.append(train_loss)
            
            Yhat_val, _ = self.forward(Xval)
            val_loss = self.loss(Yval, Yhat_val)
            self.val_losses.append(val_loss)
            
            if self.verbose and (ep == 1 or ep % 20 == 0):
                print(f'Epoch: [ {ep} / {self.epochs} ] | Training Loss: {train_loss:.2f} | Val. Loss: {val_loss:.2f}')
    
    
    
    def fit_RMSProp(self, X, Y, Xval, Yval, decay=0.99, mu=0.9):
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        Xval = Xval.astype(np.float32)
        Yval = Yval.astype(np.float32)
        
        N, D = X.shape
        n_batches = int(np.ceil(N / self.batch_size))
        
        self.train_losses = []
        self.val_losses = []
        
        eps = 1e-08
        
        for ep in range(1, self.epochs+1):
            for j in range(n_batches):
                Xbatch = X[j*n_batches:(j+1)*n_batches,:]
                Ybatch = Y[j*n_batches:(j+1)*n_batches,:]
                
                Yhat_batch, Z = self.forward(Xbatch)

                delta = Yhat_batch-Ybatch
                gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
                self.layers[-1].cacheW = decay * self.layers[-1].cacheW + (1-decay) * gradW * gradW
                self.layers[-1].cacheb = decay * self.layers[-1].cacheb + (1-decay) * gradb * gradb
                self.layers[-1].vW = mu * self.layers[-1].vW + (1-mu) * self.learning_rate * (gradW + self.reg*self.layers[-1].W) / (np.sqrt(self.layers[-1].cacheW) + eps)
                self.layers[-1].vb = mu * self.layers[-1].vb + (1-mu) * self.learning_rate * gradb / (np.sqrt(self.layers[-1].cacheb) + eps)
                self.layers[-1].W -= self.layers[-1].vW
                self.layers[-1].b -= self.layers[-1].vb
                
                for i in range(len(self.layers)-2,-1,-1):
                    gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
                    self.layers[i].cacheW = decay * self.layers[i].cacheW + (1-decay) * gradW * gradW
                    self.layers[i].cacheb = decay * self.layers[i].cacheb + (1-decay) * gradb * gradb
                    self.layers[i].vW = mu * self.layers[i].vW + (1-mu) * self.learning_rate * (gradW + self.reg*self.layers[i].W) / (np.sqrt(self.layers[i].cacheW) + eps)
                    self.layers[i].vb = mu * self.layers[i].vb + (1-mu) * self.learning_rate * gradb / (np.sqrt(self.layers[i].cacheb) + eps)
                    self.layers[i].W -= self.layers[i].vW
                    self.layers[i].b -= self.layers[i].vb

            Yhat_train, _ = self.forward(X)
            train_loss = self.loss(Y, Yhat_train)
            self.train_losses.append(train_loss)
            
            Yhat_val, _ = self.forward(Xval)
            val_loss = self.loss(Yval, Yhat_val)
            self.val_losses.append(val_loss)
            
            if self.verbose and (ep == 1 or ep % 20 == 0):
                print(f'Epoch: [ {ep} / {self.epochs} ] | Training Loss: {train_loss:.2f} | Val. Loss: {val_loss:.2f}')
    

    def fit_Adam(self, X, Y, Xval, Yval, beta1=0.9, beta2=0.999):
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        Xval = Xval.astype(np.float32)
        Yval = Yval.astype(np.float32)
        
        N, D = X.shape
        n_batches = int(np.ceil(N / self.batch_size))
        
        self.train_losses = []
        self.val_losses = []
        
        eps = 1e-08
        for ep in range(1, self.epochs+1):
            for j in range(n_batches):
                Xbatch = X[j*n_batches:(j+1)*n_batches,:]
                Ybatch = Y[j*n_batches:(j+1)*n_batches,:]
                self.t += 1
                Yhat_batch, Z = self.forward(Xbatch)

                delta = Yhat_batch-Ybatch
                gradW, gradb, delta = self.layers[-1].grad(delta, Z[-1])
                self.layers[-1].mW = beta1 * self.layers[-1].mW + (1-beta1) * gradW
                self.layers[-1].mb = beta1 * self.layers[-1].mb + (1-beta1) * gradb
                self.layers[-1].vW = beta2 * self.layers[-1].vW + (1-beta2) * gradW * gradW
                self.layers[-1].vb = beta2 * self.layers[-1].vb + (1-beta2) * gradb * gradb
                mW_hat = self.layers[-1].mW / (1-beta1**self.t)
                mb_hat = self.layers[-1].mb / (1-beta1**self.t)
                vW_hat = self.layers[-1].vW / (1-beta2**self.t)
                vb_hat = self.layers[-1].vb / (1-beta2**self.t)
                self.layers[-1].W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
                self.layers[-1].b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
                
                for i in range(len(self.layers)-2,-1,-1):
                    gradW, gradb, delta = self.layers[i].grad(delta, Z[i], Z=Z[i+1], W_next=self.layers[i+1].W)
                    self.layers[i].mW = beta1 * self.layers[i].mW + (1-beta1) * gradW
                    self.layers[i].mb = beta1 * self.layers[i].mb + (1-beta1) * gradb
                    self.layers[i].vW = beta2 * self.layers[i].vW + (1-beta2) * gradW * gradW
                    self.layers[i].vb = beta2 * self.layers[i].vb + (1-beta2) * gradb * gradb
                    mW_hat = self.layers[i].mW / (1-beta1**self.t)
                    mb_hat = self.layers[i].mb / (1-beta1**self.t)
                    vW_hat = self.layers[i].vW / (1-beta2**self.t)
                    vb_hat = self.layers[i].vb / (1-beta2**self.t)
                    self.layers[i].W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
                    self.layers[i].b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
                
                
            
            Yhat_train, _ = self.forward(X)
            train_loss = self.loss(Y, Yhat_train)
            self.train_losses.append(train_loss)
            
            Yhat_val, _ = self.forward(Xval)
            val_loss = self.loss(Yval, Yhat_val)
            self.val_losses.append(val_loss)
            
            if self.verbose and (ep == 1 or ep % 20 == 0):
                print(f'Epoch: [ {ep} / {self.epochs} ] | Training Loss: {train_loss:.2f} | Val. Loss: {val_loss:.2f}')

    
    def forward(self, X):
        Z = [X]
        for h in self.layers[:-1]:
            Z.append(h.forward(Z[-1]))
        
        Yhat = self.layers[-1].forward(Z[-1])
        return Yhat, Z

    def loss(self, Y, Yhat):
        return ((Y - Yhat)**2).mean()

    def predict(self, X):
        Yhat, _ = self.forward(X)
        return Yhat
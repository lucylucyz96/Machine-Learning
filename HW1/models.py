""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np
import math


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, lr):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class MCModel(Model):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures)
        self.num_classes = nclasses


class MCPerceptron(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        # TODO: Implement this!  
        for i in range(0,X.shape[0]):
            maximum = float("-inf")
            prediction_class=0
            #make the prediction
            for k in range(0,self.W.shape[0]):
                y_hat = X[i].dot(self.W[k])
                if(y_hat>maximum): #if >= then different, what should we use
                    prediction_class = k
                    maximum = y_hat
                    #print(prediction_class,k)
            #Updating wk:
            if (prediction_class != y[i]):
                    #print(prediction_class,y[i])
                self.W[prediction_class] = self.W[prediction_class]-(lr*X[i])
                self.W[y[i]] = self.W[y[i]]+(lr*X[i])
        #return self.W

        #raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        all_predictions = []
        for i in range(0,X.shape[0]):
            maximum = float("-inf")
            for k in range(0,self.W.shape[0]):
                y_hat = X[i].dot(self.W[k])
                if(y_hat>maximum):
                    prediction_class = k
                    maximum = y_hat
                    #print(prediction_class, maximum)
            all_predictions.append(prediction_class)
        return all_predictions
        
        raise Exception("You must implement this method!")

    def score(self,X,K):
        return X.dot(self.W[K])[0]
        #for k in range(0,self.W.shape[0]):
            #print(self.W[k])
            #y_hat = X.dot(self.W[k])


class MCLogistic(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        for i in range (0, X.shape[0]):
            logits_g = []
            for k in range (0,self.W.shape[0]):
                gk = X[i].dot(self.W[k])
                logits_g.append(gk)
            #print(logits_g)
            softmax_gk = self.softmax(logits_g)
            #print(softmax_gk)
            #updating wk
            for k1 in range(0, self.W.shape[0]):
                if k1 != y[i]:
                    self.W[k1] =self.W[k1]-((softmax_gk[k1]* X[i])*lr)
                    #print(self.W[k1])
                else:
                    self.W[k1] =self.W[k1]+ ((X[i]- (softmax_gk[k1]* X[i]))*lr)
                    #print(self.W[k1])
        #return self.W
        #raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        all_predictions = []
        for i in range(0,X.shape[0]):
            maximum = float("-inf")
            for k in range(0,self.W.shape[0]):
                y_hat = X[i].dot(self.W[k])
                if(y_hat>=maximum):
                    #print(y_hat,k)
                    prediction_class = k
                    maximum = y_hat
            all_predictions.append(prediction_class)
        #print(all_predictions)
        return all_predictions
        raise Exception("You must implement this method!")

    def softmax(self, logits):
        # TODO: Implement this!
        g_star = max(logits)
        softmax_gk = []
        down = 0
        for val in logits:
            #print(val)
            down=down+ math.exp(val-g_star)
        for val in logits:
            up =  math.exp(val-g_star)
            softmax_gk.append(up/down)
        return softmax_gk
        #raise Exception("You must implement this method!")

    def score(self,X,K):
        return X.dot(self.W[K])[0]

class OneVsAll(Model):

    def __init__(self, *, nfeatures, nclasses, model_class):
        super().__init__(nfeatures)
        self.num_classes = nclasses
        self.model_class = model_class
        self.models = [model_class(nfeatures=nfeatures, nclasses=2) for _ in range(nclasses)]

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        #for a list of lists of y consisting of 0 and 1

            #train the model to get W
        count = 0
        for k in self.models: 
            y_k = []
            for i in range (0, X.shape[0]):
                if(y[i]==count):
                    y_k.append(1)
                else:
                    y_k.append(0)
            count = count+1
            k.fit(X=X,y=y_k,lr = lr)
                #print(score)


        #raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        all_predictions = []
        for i in range(0,X.shape[0]):
            maximum = float("-inf")
            count = 0
            for k in self.models:
                y_hat = k.score(X=X[i],K=1)
                if(y_hat>maximum):
                    prediction_class = count
                    maximum = y_hat
                count = count+1
                    #print(prediction_class, maximum)
            all_predictions.append(prediction_class)
        #print(all_predictions)
        return all_predictions
        raise Exception("You must implement this method!")

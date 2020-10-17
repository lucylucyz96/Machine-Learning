import numpy as np
from collections import defaultdict
import scipy
from scipy import linalg, spatial, stats
from scipy.sparse.linalg import eigsh
from scipy.spatial import distance

# TODO: You can import anything from numpy or scipy here!

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
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

class PCA(Model):

    def __init__(self, X, target_dim):
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]
        self.target_dim = target_dim
        self.W = None

    def fit(self, X):
        # TODO: Implement!
        #print(X.shape)
        st_dev = X.std(axis=0)
        st_dev[st_dev==0]=1.0
        mean = X.mean(axis=0)

        normalized = (X-mean)/st_dev

        covariance = np.cov(np.transpose(normalized))
        eigenValues, eigenVectors = linalg.eig(covariance)
        np.absolute(eigenValues)
        idx = eigenValues.argsort()[-self.target_dim:][::-1]   
        #eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]

        #W = eigenVectors[:,:self.target_dim]
        result = np.dot(normalized,eigenVectors)
        return result
        

        #raise NotImplementedError()

class LLE(Model):

    def __init__(self, X, target_dim, lle_k):
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]

        self.target_dim = target_dim
        self.k = lle_k

    def fit(self, X):
        # TODO: Implement!

        #normalize the data
        st_dev = X.std(axis=0)
        st_dev[st_dev==0]=1.0
        mean = X.mean(axis=0)
        normalized = (X-mean)/st_dev
        X=normalized

        #k nearest neighors
        kd_tree = scipy.spatial.cKDTree(X)
        _, index = kd_tree.query(X, self.k+1)
        index = index[:,1:]
        #print(index)

        
        #Solve for reconstruction weights
        W = np.zeros((X.shape[0], X.shape[0]), dtype=np.float)
        for i in range(X.shape[0]):
            Z = np.zeros((X.shape[1],self.k),dtype=np.float)
            for k in range(self.k):
                Z[:,k] = X[index[i][k]]-X[i]
            C = np.matmul(Z.T,Z)
            reg_term = (10 ** (-3)) * np.trace(C)
            C = C+ reg_term * np.identity(C.shape[0])
            w = linalg.solve(C, np.ones((C.shape[0], 1)))
            w_sum =np.sum(w)
            for k in range(self.k):
                W[i][index[i][k]] = w[k]/w_sum
        

        #Compute embedding coordinates
        
        M = np.matmul((np.identity(self.num_x) - W).T, (np.identity(self.num_x) - W))
        eigen_val, eigen_vec = eigsh(M, k=self.target_dim+1,sigma=0.0)
        eps = np.finfo(float).eps
        eigen_vec = eigen_vec[:,eigen_val>eps]
        #print(eigen_vec.shape)

        return eigen_vec
        
        

class KNN(Model):

    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, X, y):
        self.data = X
        self.labels = y

    def predict(self, X):
        kd_tree = scipy.spatial.cKDTree(self.data)
        _, index = kd_tree.query(X, self.k)
        #index = index[:,1:]
        #print(index)
        return_label = []
        for i in index:
            label = self.most_frequent(i)
            return_label.append(label)
        return return_label
        #raise NotImplementedError()
    
    def most_frequent(self,List): 
        y_labels = [self.labels[l] for l in List]
        #print(y_labels)
        itm = scipy.stats.mode(y_labels)[0]
        #print(itm)
        return(itm) 
    

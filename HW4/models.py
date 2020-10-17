"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np

def initialize_variational_parameters(num_rows_of_image, num_cols_of_image, K):
    """ Helper function to initialize variational distributions before each E-step.
    Args:
                num_rows_of_image: Integer representing the number of rows in the image
                num_cols_of_image: Integer representing the number of columns in the image
                K: The number of latent states in the MRF
    Returns:
                q: 3-dimensional numpy matrix with shape [num_rows_of_image, num_cols_of_image, K]
     """
    q = np.random.random((num_rows_of_image, num_cols_of_image, K))
    for row_num in range(num_rows_of_image):
        for col_num in range(num_cols_of_image):
            q[row_num, col_num, :] = q[row_num, col_num, :]/sum(q[row_num ,col_num, :])
    return q

def initialize_theta_parameters(K):
    """ Helper function to initialize theta before begining of EM.
    Args:
                K: The number of latent states in the MRF
    Returns:
                mu: A numpy vector of dimension [K] representing the mean for each of the K classes
                sigma: A numpy vector of dimension [K] representing the standard deviation for each of the K classes
    """
    mu = np.zeros(K)
    sigma = np.zeros(K) + 10
    for k in range(K):
        mu[k] = np.random.randint(10,240)
    return mu, sigma


class MRF(object):
    def __init__(self, J, K, n_em_iter, n_vi_iter):
        self.J = J
        self.K = K
        self.n_em_iter = n_em_iter
        self.n_vi_iter = n_vi_iter
        self.q  = None
    def fit(self, *, X):
        """ Fit the model.
                Args:
                X: A matrix of floats with shape [num_rows_of_image, num_cols_of_image]
        """
        # TODO: Implement this!
        # Please use helper function 'initialize_theta_parameters' to initialize theta at the start of EM 
        #     Ex:  mu, sigma = initialize_theta_parameters(self.K)
        # Please use helper function 'initialize_variational_parameters' to initialize q at the start of each E step 
        #     Ex:  q = initialize_variational_parameters(X.shape[0], X.shape[1], self.K)
        #print(X.shape)
        mu,sigma = initialize_theta_parameters(self.K)
        sigma= sigma**2
        for n in range (self.n_em_iter):
            self.q = self.variational_inference(X,mu,sigma)
            mu, sigma = self.max_likelihood_estimation(X,mu,sigma)
        self.q = self.variational_inference(X,mu,sigma)


    
    def variational_inference(self,X,mu,sigma):
        q = initialize_variational_parameters(X.shape[0], X.shape[1], self.K)
        #print(q[1][0][1])
        for m in range(self.n_vi_iter):
            for row in range(X.shape[0]):
                for col in range(X.shape[1]):
                    bottom = 0
                    curr_pixel = X[row][col]
                    for k in range(self.K):
                        normal = self.normal_distribution(curr_pixel,mu[k],sigma[k]) 
                        curr_neighbors = self.get_neighbors(X.shape[0],X.shape[1],row,col,q,k)
                        bottom = bottom+normal*np.exp(curr_neighbors * self.J)
                       
                    for i in range(self.K):
                        top_normal = self.normal_distribution(curr_pixel,mu[i],sigma[i])
                        neighbors = self.get_neighbors(X.shape[0],X.shape[1],row,col,q,i)
                        top = top_normal * np.exp(neighbors * self.J)
                        q[row][col][i] = top/bottom
        return q
    
    def max_likelihood_estimation(self,X,mu,sigma):
        for k in range(self.K):
            sum_top_mu = 0
            sum_bottom = 0
            sum_top_sigma = 0
            for row in range(X.shape[0]):
                for col in range(X.shape[1]):
                    pixel = X[row][col]
                    var_approx = self.q[row][col][k]
                    sum_top_mu = sum_top_mu + pixel * var_approx
                    sum_bottom = sum_bottom + var_approx 
            mu[k] = sum_top_mu/sum_bottom
            for row in range(X.shape[0]):
                for col in range(X.shape[1]):
                    var_approx_1 = self.q[row][col][k]
                    pixel = X[row][col]
                    sum_top_sigma = sum_top_sigma+(var_approx_1*((pixel-mu[k])**2))

            sigma[k] = sum_top_sigma/sum_bottom
        return mu,sigma


    def get_neighbors(self,width,length,row,col,q,k):
        ##print(X.shape[0])
         #get neighboring pixels
        neighborings = [(row+1,col),(row-1,col),(row,col+1),(row,col-1)]
        ans = 0
        for rowN, colN in neighborings:
            if 0<=rowN<width and 0<=colN<length:
                ans+=q[rowN][colN][k]
        return ans


    def normal_distribution(self,pixel,mu,sigma):
        normal_distri = (1/(np.sqrt(2*np.pi*sigma)))*np.exp((-((pixel - mu)**2))/(2*sigma))
        return normal_distri

    def predict(self, X):
        """ Predict.
        Args:
                X: A matrix of floats with shape [num_rows_of_image, num_cols_of_image]

        Returns:
                A matrix of ints with shape [num_rows_of_image, num_cols_of_image].
                    - Each element of this matrix should be the most likely state according to the trained model for the pixel corresponding to that row and column
                    - States should be encoded as {0,..,K-1}
        """
        # TODO: Implement this!
        matrix_of_preds = []
        for row in range(X.shape[0]):
            list_of_preds=[]
            for col in range(X.shape[1]):
                maximum = float("-inf")
                max_index = -1
                ps = self.q[row][col]
                for k in range(len(ps)):
                    if (ps[k]>maximum):
                        maximum = ps[k]
                        max_index = k
                list_of_preds.append(max_index)
            matrix_of_preds.append(list_of_preds)
        matrix = np.array(matrix_of_preds)
        #print(matrix_of_preds)
        return matrix

"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np

class RegressionTree(object):
    def __init__(self, nfeatures, max_depth):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.val_list = [None]*((2**(max_depth+1)))#create an empty list of length # of features to store theta and d
    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
        """
        # TODO: Implement this!
        n = 1
        self.findVal(X,y,n)
        #print(self.val_list)

        #raise Exception("You must implement this method!")

    def findVal(self,X,y,pos):
        best_f =-1
        best_score = float('inf')-1
        best_theta = -1

        #Check for Base case 1
        if(len(y)<=1):
            self.val_list[pos]=(sum(y)/len(y))
            return 
        #Check for Base case 2
        flag = 0
        for col in X.T:
            if(len(set(col))!=1):
                flag = 1
        if(flag == 0):
            self.val_list[pos]=(sum(y)/len(y))
            return 

        #base case 3 reached max depth
        if(((2*pos)>(len(self.val_list)-1)) or ((2*pos+1)>len(self.val_list)-1)):
            self.val_list[pos]=(sum(y)/len(y))
            return 

        count = 0
        score = 0
        L = []
        R = []
        L_y = []
        R_y=[]
        for v in X.T: #Iterate through feature
            #looking for threshold 
            v = np.unique(v)
            for x in v:
                theta = x
                score = self.findScore(count,theta,X,y)
                if(score<best_score):
                    best_score = score 
                    best_theta = theta
                    best_f = count
            count = count +1

        L = self.findLeft(best_f,best_theta,X)
        R = self.findRight(best_f,best_theta,X)
        L_y=self.findLeft_y(best_f,best_theta,X,y)
        R_y = self.findRight_y(best_f,best_theta,X,y)

        L = np.array(L,dtype = 'float')
        R = np.array(R,dtype = 'float')
        L_y = np.array(L_y,dtype = 'float')
        R_y = np.array(R_y,dtype = 'float')
        self.val_list[pos]=(best_f,best_theta)

        self.findVal(L,L_y,2*pos)
        self.findVal(R,R_y,2*pos+1)

    def findLeft(self,v,theta,X):
        L_x = []
        for x in X:
            if(x[v]<theta):
                L_x.append(x)
        return L_x

    def findRight(self,v,theta,X):
        R_x = []
        for x in X:
            if(x[v]>=theta):
                R_x.append(x)
        return R_x

    def findRight_y(self,v,theta,X,y):
        yi= []
        count = 0
        for x in X:
            if(x[v]>=theta):
                yi.append(y[count])
            count = count+1

        return yi

    def findLeft_y(self,v,theta,X,y):
        yi= []
        count = 0
        for x in X:
            if(x[v]<theta):
                yi.append(y[count])
            count = count+1

        return yi

    def findScore(self,v,theta,X,y):
        L = []
        R = []
        count = 0
        for x in X:
            if(x[v]<theta):
                L.append(y[count])

            if(x[v]>=theta):
                R.append(y[count])
            count = count+1

        if(len(L)==0 or len(R)==0):
            return float('inf')


        mean_L = (1/len(L))*sum(L)
        mean_R = (1/len(R))*sum(R)

        score = 0
        sum_l = 0
        sum_r = 0

        for yi in L:
            sum_l = sum_l+(yi-mean_L)**2
        for yi in R:
            sum_r = sum_r +(yi-mean_R)**2
        score = sum_l+sum_r

        return score



    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this
        predictions = []
        for x in X:
            count = 1
            pos = 1
            flag = 0
            while(count<self.max_depth+1):
                #print(self.val_list[pos])
                feature = self.val_list[pos][0]
                threshold = self.val_list[pos][1]
                if(x[feature]<threshold):
                    if(isinstance(self.val_list[2*pos],float)==True): #check for leaf node
                        predictions.append(self.val_list[2*pos])
                        flag = 1
                        break
                    else:
                        pos = 2*pos
                if(x[feature]>=threshold):
                    if(isinstance(self.val_list[2*pos+1],float)==True):
                        predictions.append(self.val_list[2*pos+1])
                        flag = 1
                        break
                    else:
                        pos = 2*pos+1
                count = count+1
            if(flag == 0):
                predictions.append(self.val_list[pos])
        return predictions


        #raise Exception("You must implement this method!")



class GradientBoostedRegressionTree(object):
    def __init__(self, nfeatures, max_depth, n_estimators, regularization_parameter):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.regularization_parameter = regularization_parameter
        self.models = []
        self.initial = None
    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """
        # Initialize F_0 to all averages
        f0_value= sum(y)/len(y)
        F_0 = [f0_value]*len(y) 
        #print(F_0)
        self.initial = f0_value
        #Iterate through m times
        for m in range(0,self.n_estimators):
            #Get the residuals
            g_values = []
            count = 0
            for x in X:
                gi = y[count]-F_0[count]
                g_values.append(gi)
                count = count+1

            #train a RegressionTree
            h_m = RegressionTree(self.num_input_features,self.max_depth)
            h_m.fit(X=X,y=g_values)
            self.models.append(h_m)

            #get the residuals and update our predictions
            predictions = h_m.predict(X)
            #print(predictions)
            for vals in range(0,len(F_0)):
                F_0[vals] = F_0[vals]+ self.regularization_parameter * predictions[vals]

        #print(self.models)
        

        #raise Exception("You must implement this method!")

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        prediction =[self.initial]*len(X)
        #predictions = np.zeros(len(X)) + self.initial
        for m in self.models:
            predict = m.predict(X)
            predict = np.asarray(predict)
            predict = predict * self.regularization_parameter
            prediction = prediction+predict
        #print(prediction)
        return prediction

        #raise Exception("You must implement this method!")

import numpy as np
from models.linear_loss import * 

class LinearRegression(object):

    def __init__(self):  
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, verbose=False):
           
        num_train, dim = X.shape
        if self.W is None:
          # lazily initialize W
          self.W = 0.001 * np.random.randn(dim, 1)

        # Run gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            loss, grad = self.loss(X, y, reg)
            loss_history.append(loss)

          # perform parameter update
          #########################################################################
          # TODO:                                                                 #
          # Update the weights using the gradient and the learning rate.          #
          #########################################################################
            self.W = self.W-learning_rate*grad
          #########################################################################
          #                       END OF YOUR CODE                                #
          #########################################################################
        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history
    
    def predict(self, X):
    
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        y_pred = np.dot(X,self.W)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X, y, reg):
        """
        Define the loss function with linear_loss_naive or linear_loss_vectorized
        """
        #########################################################################
        # TODO:                                                                 #
        # Calculate the loss value and gradient matrix with linear_loss_..      #
        #########################################################################
        loss, grad = linear_loss_vectorized(self.W,X,y,reg)
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        return loss, grad

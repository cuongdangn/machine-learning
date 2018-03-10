import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D,) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = y.size
    D = X[0].size
    # print(N," ",D)
    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(N) :
        temp_sum = 0
        for j in range(D) :
            temp_sum += X[i][j]*W[j]
        for j in range(D) :
            dW[j] += (temp_sum-y[i])*X[i][j]
        loss = loss + np.power(temp_sum - y[i], 2)
    loss = loss / (2*N) + reg
    
    
    #dW1 = np.dot(X.transpose(), np.dot(X,W)-y)
    dW = dW / N
   
    #print(dW1-dW/N)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
   # print("dW.shape 1",dW.shape)
   # print(dW/N)
    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = y.size
 
    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    loss = np.sum((np.dot(X,W)-y)**2)/(2*N) + reg
    tmp = np.dot(X,W) -y
 #   tmp = tmp[:,np.newaxis]
  #  print("y.transpose ",y.transpose().shape)
  #  print("tmp.shape ", tmp.shape)
  #  print(W)
    dW = np.dot(X.transpose(), tmp)
    dW = dW / N
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    #print(dW)
  #  print("W,.shape ",W.shape)
  #  print("y.shape ",y.shape)
  #  print("X.transpose() . shape",X.transpose().shape)
  #  print("dW.shape ",dW.shape)
    return loss, dW
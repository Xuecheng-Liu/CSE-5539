#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# AU 19 CSE 5539-0010 "Social Media and Text Analysis" Homework #2  
# Wei Xu, The Ohio State University   
# 
# In this assignment, we will walk you through the process of :
# 
# - implementing logistic regression from scratch 
# - and applying it to a real-world problem that predicts whether a student will be admitted to a university. 
# 

# <div class="alert alert-danger">
# IMPORTANG: In this assignment, except Numpy and Matplotlib, no other external Python packages are allowed. Scipy is used in gradient checking, though, it is not allowed elsewhere. 
# </div>

# **Honor Code:** I hereby agree to abide the Ohio State University's Code of Student Conduct, promise that the submitted assignment is my own work, and understand that my code is subject to plagiarism test.
# 
# **Signature**: *(double click on this block and type your name here)*

# ## 0. Importing Numpy and Matplotlib [Code provided - do not change]

# In[1]:


import sys

# Check what version of Python is running
print (sys.version)


# You will need to have Numpy installed for the right version of Python. Most likely, you are using Python 3.6 in this Jupyter Notebook; then you may [install Numpy accordingly](https://stackoverflow.com/questions/37933978/install-numpy-in-python-3-4-4-and-linux). For example, installng Numpy via pip by using the command line "sudo python3.6 -m pip install numpy". If failed, you may need to update pip first by "python3.6 -m pip install --upgrade pip".

# In[2]:


# Run some setup code for this notebook. Don't modify anything in this cell.

import random
import numpy as np
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# reload external python modules;
# http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## 1. Visualizing the Data  [Code provided - no need to change]
# 
# The provided dataset contains applicants' scores on two exams and the admission decisons. 
# 
# [Matplotlib](http://matplotlib.org/users/pyplot_tutorial.html) is a Python package for data visualization. Suppose you are using Python 3.6, you can install by first use command line "brew install freetype", then "sudo python3.6 -m pip install matplotlib". 
# 

# In[3]:


#load the dataset
data = np.loadtxt('hw2_data.txt', delimiter=',')

train_X = data[:, 0:2]
train_y = data[:, 2]

# Get the number of training examples and the number of features
m_samples, n_features = train_X.shape
print ("# of training examples = ", m_samples)
print ("# of features = ", n_features)

pos = np.where(train_y == 1)
neg = np.where(train_y == 0)
plt.scatter(train_X[pos, 0], train_X[pos, 1], marker='o', c='b')
plt.scatter(train_X[neg, 0], train_X[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Not Admitted', 'Admitted'])
plt.show()


# ## 2. Cost Function [5 points]
# You're going to first implement the sigmoid function, then the cost function for (binary) logistic regression. 
# 
# The sigmoid function is defined as $sigmoid(\mathbf{z}) = \frac{1}{1+{e^{-\mathbf{z}}}}$.
# 
# Note that, you are asked to use the _numpy_ package for vector and matrix operations in order to ensure the __efficiency of the code__. 

# In[7]:


def sigmoid(z):
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################
    
    ### YOUR CODE HERE: be careful of the potential underflow or overflow here
    
    

    s = 1/(1+np.exp(-z))
    
    
    
    ### END YOUR CODE
    
    return s

# Check your sigmoid implementation
z = np.array([[1, 2], [-1, -2]])
f = sigmoid(z)
print ("=== For autograder ===")
print (f)


# In[40]:


def cost_function(theta, X, y):
    """ The cost function for logistic regression """
    #####################################################################################
    # Compute the cost given the current parameter theta on the training data set (X, y)#
    #####################################################################################
     
    ### YOUR CODE HERE
    
    yhat = sigmoid(np.dot(X,theta))
    log_yhat = np.log(yhat)
    log_one_minus_yhat = np.log(1-yhat)
    cost = -1.0/float(X.shape[0]) * (np.sum(y * log_yhat) + np.sum((1-y) * log_one_minus_yhat))

    
    
    
    
    
    ### END YOUR CODE
    
    return cost

# Check your cost function implementation

t_X = np.array([[1, 2], [-1, -2]])
t_y = np.array([0, 1])
t_theta1 = np.array([-10, 10])
t_theta2 = np.array([10, -10])
t_c1 = cost_function(t_theta1, t_X, t_y)
t_c2 = cost_function(t_theta2, t_X, t_y)
print ("=== For autograder ===")
print (t_c1)
print (t_c2)


# ## 3. Gradient Computation [5 points]
# 
# Implement the gradient computations for logistic regression. 

# In[41]:


def gradient_update(theta, X, y):
    """ The gradient update for logistic regression"""
    ###############################
    # Compute the gradient update #
    ###############################
    
    ### YOUR CODE HERE
    m_samples = X.shape[0]
    
    yhat = sigmoid(np.dot(X,theta))
    
    grad = np.sum((yhat - y).reshape(m_samples,1) * X,0)
    
    
    
    

    ### END YOUR CODE
    
    grad = grad / m_samples  
    
    return grad

# Check your gradient computation implementation
t_X = np.array([[1, 2], [-1, -2]])
t_y = np.array([0, 1])
t_theta1 = np.array([-10, 10])
t_theta2 = np.array([10, -10])
t_g1 = gradient_update(t_theta1, t_X, t_y)
t_g2 = gradient_update(t_theta2, t_X, t_y)
print ("=== For autograder ===")
print (t_g1)
print (t_g2)


# ## 4. Gradient Checking [Code provided. Bonus 5 points if implemented from scratch]
# You can use the code provided below to check the gradient of your logistic regression functions. Alternatively, you can implementing the gradient checking from scratch by yourself (bonus 10 points). 
# 
# [Gradient checking](http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/) is an important technique for debugging the gradient computation. Logistic regression is a relatively simple algorithm where it is straightforward to derive and implement its cost function and gradient computation. For more complex models, the gradient computaitn can be notoriously difficulty to debug and get right. Sometimes a subtly buggy implementation will manage to learn something that can look surprisingly reasonable, while performing less well than a correct implementation. Thus, even with a buggy implementation, it may not at all be apparent that anything is amiss. 

# In[29]:


# Check your gradient computation implementation
t_samples, t_features = 100, 10
t_X = np.random.randn(t_samples, t_features)
t_y = np.random.randint(2, size=t_samples) 
t_theta = np.random.randn(t_features)

from scipy import optimize
print ("=== For autograder ===")
print('Output of check_grad: %s' % optimize.check_grad(cost_function, gradient_update, t_theta, t_X, t_y))


# ## 5. Gradient Descent  and Decision Boundary  [10 points]
# 
# Implement the batch gradient decent algorithm for logistic regression. For every 'print_iterations' number of iterations, also visualize the decision boundary and obeserve how it changes during the training.
# 
# Note that, you will need to carefully choose the learning rate and the total number of iterations, especially given that the starter code does not include feature scaling (e.g., scale each feature by its maximum absolute value to convert feature value to [-1,1] range -- in order to make this homework simple and easier for you to write code to visualize. 

# In[69]:


def gradient_descent(theta, X, y, alpha, max_iterations, print_iterations):
    """ Batch gradient descent algorithm """
    #################################################################
    # Update the parameter 'theta' iteratively to minimize the cost #
    # Also visualize the decision boundary during learning          #
    #################################################################
 
    alpha *= m_samples
    iteration = 0
    
    ### YOUR CODE HERE: 
    
    ones = np.ones((m_samples, 1), dtype=np.float)
    X = np.concatenate((X, ones), axis = 1)
    X_norm = X /np.max(X, axis = 0)
      
    ### END YOUR CODE
    
    
    
    while(iteration < max_iterations):
        iteration += 1
        
        ### YOUR CODE HERE: simultaneous update of partial gradients
        gradient = gradient_update(theta, X_norm, y)
        theta -= alpha * gradient
        ### END YOUR CODE

        
        # For every 25 iterations
        if iteration % print_iterations == 0 or iteration == 1:
            cost = 0
            
            ### YOUR CODE HERE: calculate the cost
            ### IMPORTANT: The cost function is guaranteed to decrease after 
            ## every iteration of the gradient descent algorithm.
            cost = cost_function(theta, X_norm, y)

            ### END YOUR CODE
            
            
            print ("[ Iteration", iteration, "]", "cost =", cost)
            plt.rcParams['figure.figsize'] = (5, 4)
            plt.xlim([20,110])
            plt.ylim([20,110])
            
            pos = np.where(y == 1)
            neg = np.where(y == 0)
            
            plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
            plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
            plt.xlabel('Exam 1 score')
            plt.ylabel('Exam 2 score')
            plt.legend(['Not Admitted', 'Admitted'])
            t = np.arange(10, 100, 0.1)
            
            
            ### YOUR CODE HERE: plot the decision boundary
            
            y_t = -(theta[0] * t + theta[2]) / theta[1]
            plt.plot(t, y_t, color='red', label='decision boundary');
            
            ### END YOUR CODE 
            #plt.show()
               
    return theta


### YOUR CODE HERE: initialize the parameters 'theta' to random values; 
### And set up learning rate, number of max iterations, number of iterations for printing intermedia outputs
initial_theta = np.zeros(train_X.shape[1]+1, dtype=np.float)
alpha_test = 0.05
max_iter = 1000
print_iter = 100
    
    

    
    
### END YOUR CODE


learned_theta = gradient_descent(initial_theta, train_X, train_y, alpha_test, max_iter, print_iter)


# ### 6. Predicting [5 points]
# Now that you learned the parameters of the model, you can use the model to prdict whether a particular student will be admited. 

# In[66]:


def predict(theta, X):
    """ Predict whether the label is 0 or 1 using learned logistic regression parameters """

    ### YOUR CODE HERE:
    
    X = np.array(X)
    m_samples = X.shape[0]
    ones = np.ones((m_samples, 1), dtype=np.float)
    X = np.concatenate((X, ones), axis = 1)
    X_norm = X /np.max(X, axis = 0)
    probabilities = sigmoid(X_norm.dot(theta))
    predicted_labels = np.array([1 if e >= 0.5 else 0 for e in probabilities])  
    
    
    
    
    
    
    
    
    ### END YOUR CODE
    
    
    ## convert an array of booleans 'predicted_labels' into an array of 0 or 1 intergers
    return probabilities, 1*predicted_labels 

# Check your predication function implementation
t_X1 = np.array([[90, 90]])
t_X2 = np.array([[50, 60]])
t_X3 = np.array([[10, 50]])
print ("=== For autograder ===")
print (predict(learned_theta, t_X1))
print (predict(learned_theta, t_X2))
print (predict(learned_theta, t_X3))

# Computer accuracy on the training dateset 
t_prob, t_label = predict(learned_theta, train_X)
t_precision = t_label[np.where(t_label == train_y)].size / float(train_y.size) * 100
print ("=== For autograder ===")
print('Accuracy on the training set: %s%%' % round(t_precision,2))


# ### 7. Submit Your Homework
# This is the end. Congratulations! 
# 
# Now, follow the steps below to submit your homework in [Carmen](https://carmen.osu.edu/):
# 
# 1. rename this ipynb file to 'hw2_yourdotid.ipynb' 
# 2. click on the menu 'File' --> 'Download as' --> 'Python (.py)'
# 3. pack both the above 'hw2_yourdotid.ipynb' file and the 'hw2_yourdotid.py' file into a zip file 'hw2_yourdotid.zip'
# 4. upload the zip file 'hw2_yourdotid.zip' in Carmen

# In[ ]:





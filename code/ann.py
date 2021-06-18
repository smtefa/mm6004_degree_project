# -*- coding: utf-8 -*-
"""
@author: Stefan Miletic
"""

import qn
import cupy as cp
from timeit import default_timer as timer

class ffnn():
    def __init__(self, lay_sizes, act_funs):
        '''
        The feed-forward network architecture is given by 'lay_sizes'. The
        sizes of the hidden layers together with the output layer are given
        here.
        
        If we for example want to construct a network with a single hidden
        layer with 7 neurons and a single output, we would give
        lay_sizes=[7, 1]. We do not give the size of the input layer, it is
        handled in the 'forward' method.
        
        Similarly, the activation functions for the layers described in
        'lay_sizes' are given in 'act_funs'. Possible choices include
        "sigmoid" and "id".

        Parameters
        ----------
        lay_sizes : list of positive integers
            Network architecture, excluding the input size.
        act_funs : list of strings
            Activation functions for each layer.

        Returns
        -------
        None.

        '''
        self.lay_sizes = lay_sizes
        self.act_funs = act_funs
               
    def sigmoid(self):
        '''
        Sigmoidal activation function using hyperbolic tangent.

        Returns
        -------
        float
            Activated input neuron.

        '''
        return cp.tanh(self)
    
    def forward(self, x, w):
        '''
        Passes the input vector 'x' forward through a feed-forward network
        built using the weight vector provided with 'w'. The network
        architecture is given by 'lay_sizes' and the activation functions are
        given by 'act_funs' in the class constructor.
        
        If we for example wish to build a network with 5 input values, 7
        hidden neurons and a single output value, where the hidden layer is
        activated using the sigmoid function and the output is unactivated, we
        would have to provide lay_sizes=[7, 1] and act_funs=["sigmoid", "id"].
        The size of the input layer is handled automatically when 'x' is
        passed.

        Parameters
        ----------
        x : cupy.ndarray
            Input vector.
        w : cupy.ndarray
            Weight vector.

        Returns
        -------
        output : cupy.ndarray
            Output vector.

        '''
        self.input = cp.asarray(x)
        self.w = w
        self.layers = list()
        w_len = (len(self.input) + 1)*self.lay_sizes[0] + sum([
            (self.lay_sizes[i] + 1)*self.lay_sizes[i+1]
            for i in range(len(self.lay_sizes) - 1)
            ])
        assert len(self.w) == w_len
        
        output = self.input
        shift = 0
        prev_lay = None
        for lay_size, act_fun in zip(self.lay_sizes, self.act_funs):               
            if act_fun == "sigmoid":
                output = cp.append(output, 1)
                prev_lay = len(output)
                output = cp.array([
                    cp.dot(
                        output,
                        cp.array(self.w[k*prev_lay+shift:(k+1)*prev_lay+shift])
                        ) for k in range(lay_size)])
                self.layers.append(output)
                output = ffnn.sigmoid(output)
                self.layers.append(output)
                
            elif act_fun == "id":
                output = cp.append(output, 1)
                prev_lay = len(output)
                output = cp.array([
                    cp.dot(
                        output,
                        cp.array(self.w[k*prev_lay+shift:(k+1)*prev_lay+shift])
                        ) for k in range(lay_size)])
                self.layers.append(output)
                
            shift += prev_lay*lay_size
 
        return output
    
    def batch_loss(self, w, x, t):
        '''
        Passes the input 'x' through a network weighted by 'w' and calculates
        the mean squared error (MSE) against the target vector 't'.

        Parameters
        ----------
        w : cupy.ndarray
            Weight vector.
        x : cupy.ndarray
            Input vector.
        t : cupy.ndarray
            Target vector.

        Returns
        -------
        loss : float
            MSE.

        '''
        loss = 0.5*(self.forward(x, w) - t)**2
        
        return loss
    
    def batch_grad(self, w, x, t):
        '''
        Calculates the gradient of a simple (one hidden layer) network with
        respect to weights at 'w'. The activation functions must be 'sigmoid'
        and 'id' at hidden and output layers respectively.
        
        There is a worked example of this formula in "C. M. Bishop, Pattern
        recognition and machine learning, springer, 2006, Chapter 5.3.2".
        

        Parameters
        ----------
        w : cupy.ndarray
            Weight vector.
        x : cupy.ndarray
            Input vector.
        t : cupy.ndarray
            Target vector.

        Returns
        -------
        grad : cupy.ndarray
            Network's gradient with respect to weights evaluated at 'w'.

        '''
        assert len(self.lay_sizes) == 2
        x = cp.array(x)
        ffnn.forward(self, x, w)
        a, z, y = self.layers
        in_size = len(x)
        hid_size = self.lay_sizes[0]
        out_size = len(y)
        w_hid = w[(len(x) + 1)*self.lay_sizes[0]:]

        delta_out = y-t
        delta_hid = lambda j: (1 - z[j]**2)*sum(
            [w_hid[k*hid_size:(k+1)*hid_size][j]*delta_out
             for k in range(len(y))
             ])
        
        x = cp.append(x, 1)
        z = cp.append(z, 1)
        l1_partials, l2_partials = list(), list()
        for j in range(hid_size):
            for i in range(in_size + 1):
                l1_partials.append(delta_hid(j)*x[i])              
        for k in range(out_size):
            for j in range(hid_size + 1):
                l2_partials.append(delta_out*z[j])
                
        grad = cp.array(l1_partials + l2_partials).T
        
        return grad
   
    def train(self, method, params, train_set, batch_size, epochs):
        '''
        Trains a network model given a method with its parameters, training
        set, batch size and number of epochs.
        
        Valid values include method="BFGS", "NAQ", "L-BFGS" or "SL-BFGS"
        and params=[c_armijo, rho_armijo], [mu, c_armijo, rho_armijo],
        [m, c_armijo, rho_armijo] or [m, r, c_armijo, rho_armijo] respectively.
        

        Parameters
        ----------
        method : str
            One of "BFGS", "NAQ", "L-BFGS" or "SL-BFGS".
        params : list
            List of parameters described above for some method.
        train_set : cupy.ndarray
            Cupy array of (observed, target) tuples.
        batch_size : int
            Batch size. Can be the whole training set.
        epochs : int
            Number of epochs the training will repeat over the training set.

        Returns
        -------
        w : cupy.ndarray
            Output weight vector for the trained network.

        '''
        train_loader = cp.array([train_set[i:i+batch_size]
                        for i in range(0, len(train_set), batch_size)])
        w_len = (len(train_set[0][0]) + 1)*self.lay_sizes[0] + sum([
            (self.lay_sizes[i] + 1)*self.lay_sizes[i+1]
            for i in range(len(self.lay_sizes) - 1)])
        
        #w = cp.random.rand(w_len)
        w = cp.random.uniform(-0.5, 0.5, size=(1, w_len))[0]
        qn_opt = qn.Optimizers(tol=1e-3)

        if method == "BFGS":
            curv = cp.eye(len(w))
            c_armijo, rho_armijo = params
            optimizer = lambda x, f, Df, H: qn_opt.bfgs(x, f, Df, H,
                                                        c_armijo=c_armijo,
                                                        rho_armijo=rho_armijo)
        elif method == "L-BFGS":
            curv = list()
            k = 0
            m, c_armijo, rho_armijo = params
            optimizer = lambda x, f, Df, SY, k: qn_opt.lbfgs(x, f, Df, SY,
                                                             m=m,
                                                             k=k,
                                                             c_armijo=c_armijo,
                                                             rho_armijo=rho_armijo)
        elif method == "SL-BFGS":
            m, r, c_armijo, rho_armijo = params
            optimizer = lambda x, f, Df: qn_opt.slbfgs(x, f, Df, m=m, r=r,
                                                       c_armijo=c_armijo,
                                                       rho_armijo=rho_armijo)
        elif method == "NAQ":
            curv = cp.eye(len(w))
            p = cp.zeros(len(w))
            mu, c_armijo, rho_armijo = params
            optimizer = lambda x, f, Df, p, H: qn_opt.naq(x, f, Df, p, H,
                                                          mu=mu,
                                                          c_armijo=c_armijo,
                                                          rho_armijo=rho_armijo)
 
        losses, times = list(), list()
        print(f'Training the model using {method} with {params}\n')
        for epoch in range(epochs):
            cp.random.shuffle(train_loader)
            start = timer()
            for i, batch in enumerate(train_loader):
                E = lambda w: cp.sum(cp.array([
                    ffnn.batch_loss(self, w, x, t) for (x, t) in batch
                    ]))/len(batch)
                DE = lambda w: cp.sum(cp.array([
                    ffnn.batch_grad(self, w, x, t)[0] for (x, t) in batch
                    ]), axis=0)/len(batch)
                
                try:
                    if method == "SL-BFGS":
                        w = optimizer(w, E, DE)
                    elif method == "L-BFGS":
                        w, curv, k = optimizer(w, E, DE, curv, k)
                    elif method == "NAQ":
                        w, p, curv = optimizer(w, E, DE, p, curv)
                    else:
                        w, curv = optimizer(w, E, DE, curv)
                except:
                    break
                
                time = timer() - start
                print(f'Epoch [{epoch+1}/{epochs}], Step [{(i+1)*len(batch)}/{len(train_set)}], Loss: {E(w):.4f}')
                losses.append(E(w))
                times.append(time)
            
        print(f'\nAverage training loss: {sum(losses)/len(losses):.2f}')
        print(f'Best training loss: {min(losses):.2f}')
        print(f'Worst training loss: {max(losses):.2f}')
        
        print(f'\nAverage training time: {sum(times)/len(times):.2f} s')
        print(f'Best training time: {min(times):.2f} s')
        print(f'Worst training time: {max(times):.2f} s')
        
        return w
    
    def test(self, w, test_set, batch_size, epochs):
        '''
        Tests a network's performance on a test set

        Parameters
        ----------
        w : cupy.ndarray
            Weight vector of a network to be tested.
        test_set : cupy.ndarray
            Cupy array of testing tuples.
        batch_size : int
            Batch size. Can be the whole testing set.
        epochs : int
            Number of epochs the testing will repeat over the testing set.

        Returns
        -------
        None.

        '''
        test_loader = cp.array([test_set[i:i+batch_size]
                        for i in range(0, len(test_set), batch_size)])
        
        losses = list()
        print('Testing the model...')
        for epoch in range(epochs):
            cp.random.shuffle(test_loader)
            for i, batch in enumerate(test_loader):
                E = lambda w: cp.sum(cp.array([
                    ffnn.batch_loss(self, w, x, t) for (x, t) in batch
                    ]))/len(batch)
                
                losses.append(E(w))
                
            print(f'\nAverage testing loss: {sum(losses)/len(losses):.2f}')
            print(f'Best testing loss: {min(losses):.2f}')
            print(f'Worst testing loss: {max(losses):.2f}')


    
    
        

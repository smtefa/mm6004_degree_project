# -*- coding: utf-8 -*-
"""
@author: Stefan Miletic
"""

import utils
import cupy as cp
from cupy.linalg import norm

class Optimizers():
    def __init__(self, tol):
        '''
        Initialize an Optimizer class object with convergence tolerance 'tol'.

        Parameters
        ----------
        tol : float
            Convergence tolerance.

        Returns
        -------
        None.

        '''
        self.tol = tol
        
    def bfgs(self, x, f, Df, H, c_armijo, rho_armijo):
        '''
        Computes one step of optimization using the BFGS scheme.

        Parameters
        ----------
        x : cupy.ndarray
            Current iterate.
        f : function
            Objective function.
        Df : function
            Gradient of the objective function.
        H : cupy.ndarray
            Inverse Hessian of the objective function.
        c_armijo : float
            Line search parameter 0 < c < 1.
        rho_armijo : float
            Line search parameter 0 < rho < 1.

        Returns
        -------
        x : cupy.ndarray
            New iterate.
        H : cupy.ndarray
            New inverse Hessian.

        '''
        I = cp.eye(len(H))
        
        if norm(Df(x)) > self.tol:
            p = -H.dot(Df(x))
            alp = 1
            
            while f(x + alp*p) - f(x) - c_armijo*alp*cp.dot(Df(x), p) > 1e-5:
                alp *= rho_armijo

            x_new = x + alp*p
            s = x_new - x
            y = Df(x_new) - Df(x)
            
            rho = 1/cp.dot(y, s)
            V = I - rho*cp.outer(y, s)  
            H = V.T.dot(H).dot(V) + rho*cp.outer(s, s.T)
            
            x = x_new
            
            return x, H
        
        else:
            print("Gradient norm has reached the tolerance.")

    
    def naq(self, x, f, Df, p, H, mu, c_armijo, rho_armijo):
        '''
        Computes one step of optimization using the NAQ scheme.

        Parameters
        ----------
        x : cupy.ndarray
            Current iterate.
        f : function
            Objective function.
        Df : function
            Gradient of the objective function.
        p : cupy.ndarray
            Current search direction.
        H : cupy.ndarray
            Inverse Hessian of the objective function.
        mu : float
            NAQ parameter 0 < mu < 1.
        c_armijo : float
            Line search parameter 0 < c < 1.
        rho_armijo : float
            Line search parameter 0 < rho < 1.

        Returns
        -------
        x : cupy.ndarray
            New iterate.
        p : cupy.ndarray
            New search direction.
        H : cupy.ndarray
            New inveser Hessian.

        '''
        I = cp.eye(len(H))
        
        if norm(Df(x)) > self.tol:
            x_acc = x + mu*p
            v = -H.dot(Df(x_acc))
            alp = 1
            
            while f(x_acc + alp*v) - f(x_acc) - c_armijo*alp*cp.dot(
                    Df(x_acc), v) > 1e-5:
                alp *= rho_armijo
                
            p_new = mu*p + alp*v
            x_new = x + p_new
            s = x_new - x_acc
            y = Df(x_new) - Df(x_acc)
        
            rho = 1/cp.dot(y, s)
            V = I - rho*cp.outer(y, s)
            H = V.T.dot(H).dot(V) + rho*cp.outer(s, s.T)
            
            p = p_new
            x = x_new

            return x, p, H
        
        else:
            print("Gradient norm has reached the tolerance.")
    
    def lbfgs(self, x, f, Df, SY, m, k, c_armijo, rho_armijo):
        '''
        Computes one step of optimization using the L-BFGS scheme.

        Parameters
        ----------
        x : cupy.ndarray
            Current iterate.
        f : function
            Objective function.
        Df : function
            Gradient of the objective function.
        SY : list
            List of curvature tuples (s, y).
        m : int
            Memory size - number of curvature pairs to store.
        k : int
            Current iteration/epoch number.
        c_armijo : float
            Line search parameter 0 < c < 1.
        rho_armijo : float
            Line search parameter 0 < rho < 1.

        Returns
        -------
        x : cupy.ndarray
            New iterate.
        SY : list
            Updated list of most recent curvature pairs.
        k : int
            Next iteration number.

        '''
        if len(SY) == 0:
            SY.append((x, Df(x)))
            
        if norm(Df(x)) > self.tol:       
            alp = 1
            p = -utils.two_loop(Dfx=Df(x), SY=SY)
            
            while f(x + alp*p) - f(x) - c_armijo*alp*cp.dot(Df(x), p) > 1e-5:
                alp *= rho_armijo
                
            if k > m:
                SY.pop(0)
    
            x_new = x + alp*p
            s = x_new - x
            y = Df(x_new) - Df(x)
            SY.append((s, y))
            
            x = x_new
            k += 1
            
            return x, SY, k
        
        else:
            print("Gradient norm has reached the tolerance.")
    
    def slbfgs(self, x, f, Df, m, r, c_armijo, rho_armijo):
        '''
        Computes one step of optimization using the SL-BFGS scheme.

        Parameters
        ----------
        x : cupy.ndarray
            Current iterate.
        f : function
            Objective function.
        Df : function
            Gradient of the objective function.
        m : int
            Memory size - number of curvature pairs to store.
        r : float
            SL-BFGS parameter r - sampling radius.
        c_armijo : float
            Line search parameter 0 < c < 1.
        rho_armijo : float
            Line search parameter 0 < rho < 1.

        Returns
        -------
        x : cupy.ndarray
            New iterate.

        '''
            
        if norm(Df(x)) > self.tol:    
            alp = 1
            SY = utils.sample_pairs(w=x, Df=Df, m=m, r=r)
            p = -utils.two_loop(Dfx=Df(x), SY=SY)
            
            while f(x + alp*p) - f(x) - c_armijo*alp*cp.dot(Df(x), p) > 1e-5:
                alp *= rho_armijo               
    
            x_new = x + alp*p        
            x = x_new
            
            return x
        
        else:
            print("Gradient norm has reached the tolerance.")
           
        
        
        
        

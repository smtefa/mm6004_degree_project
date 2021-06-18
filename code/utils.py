# -*- coding: utf-8 -*-
"""
@author: Stefan Miletic
"""

import cupy as cp
        
def two_loop(Dfx, SY):
    '''
    Two-loop recursion discussed in "J. Nocedal and S. Wright, Numerical
    optimization, Springer Science & Business Media, 2006, p.178".

    Parameters
    ----------
    Dfx : function
        Gradient of the objective function evaluated at x.
    SY : list
        List of most recent curvature pairs.

    Returns
    -------
    r : cupy.ndarray
        r = HfDf.

    '''
    q = Dfx
    
    for s, y in reversed(SY):
        rho = 1/cp.dot(y, s)

        alp = rho*cp.dot(s, q)
        q -= alp*y
    # Suggested initial H_0, p.178.
    gamma = cp.dot(SY[-1][0], SY[-1][1])/cp.dot(SY[-1][1], SY[-1][1])
    H_0 = gamma*cp.eye(len(q))
    r = H_0.dot(q)
    for s, y in reversed(SY):
        rho = 1/cp.dot(y.T, s)
        
        beta = rho*cp.dot(y.T, r)
        r += (alp - beta)*s

    return r

def sample_pairs(w, Df, m, r):
    '''
    Sample curvature pairs for the SL-BFGS method using Option I in
    "A. S. Berahas, M. Jahani, and M. Takac, Quasi-newton methods for
    deep learning: Forget the past, just sample, arXiv preprint
    arXiv:1901.09997, (2019), Algorithm 1".

    Parameters
    ----------
    w : cupy.ndarray
        Weight vector.
    Df : function
        Gradient of the objective function..
    m : int
        Memory size - number of curvature pairs to compute.
    r : float
        SL-BFGS parameter r - sampling radius.

    Returns
    -------
    SY : list
        List of recently sampled m curvature pairs.

    '''
    SY = list()
    
    for i in range(m):
        sigma = cp.random.rand(len(w))
        w_new = w + r*sigma
        
        s = w - w_new
        y = Df(w) - Df(w_new)
        
        SY.append((s, y))
        
    return SY




     

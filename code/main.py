# -*- coding: utf-8 -*-
"""
@author: Stefan Miletic
"""

import ann
import cupy as cp

if __name__ == '__main__':
    f1 = lambda x: 1 + (x + 2*x**2)*cp.sin(-x**2)
    
    nn1 = ann.ffnn(lay_sizes=[7,1], act_funs=["sigmoid", "id"])

    print('\nGenerating training and testing sets...')
    
    # Generate sets
    Tr1_in = cp.array([x for x in cp.linspace(-4, 4, 400)])
    cp.random.shuffle(Tr1_in)
    Tr1_out = cp.array([f1(x) for x in Tr1_in])
    Tr1_norm_in = cp.array([(x - cp.min(Tr1_in))/(cp.max(Tr1_in)
                             - cp.min(Tr1_in)) for x in Tr1_in])
    Tr1_norm_out = cp.array([(y - cp.min(Tr1_out))/(cp.max(Tr1_out)
                              - cp.min(Tr1_out)) for y in Tr1_out])
    Tr1 = [(cp.array([x]), cp.array([y]))
           for x, y in zip(Tr1_norm_in, Tr1_norm_out)]

    Te1_in = cp.array([x for x in cp.random.uniform(-4, 4, size=(10000,))])
    Te1_out = cp.array([f1(x) for x in Te1_in])
    Te1_norm_in = cp.array([(x - cp.min(Te1_in))/(cp.max(Te1_in)
                             - cp.min(Te1_in)) for x in Te1_in])
    Te1_norm_out = cp.array([(y - cp.min(Te1_out))/(cp.max(Te1_out)
                              - cp.min(Te1_out)) for y in Te1_out])
    Te1 = [(cp.array([x]), cp.array([y]))
           for x, y in zip(Te1_norm_in, Te1_norm_out)]
    
    # Training/testing
    
    '''
    Viable choices for method are: "BFGS", "L-BFGS", "SL-BFGS" and "NAQ".
    When the method is chosen, set params accordingly:
        for BFGS, choose params=[c_armijo, rho_armijo],
        for NAQ, choose params=[mu, c_armijo, rho_armijo],
        for L-BFGS, choose params=[m, c_armijo, rho_armijo],
        for SL-BFGS, choose params=[m, r, c_armijo, rho_armijo].
    '''
    w = nn1.train(method="BFGS", params=[1e-3, 1e-3],
                  train_set=Tr1, batch_size=400, epochs=10)
    nn1.test(w=w, test_set=Te1, batch_size=400, epochs=1)
    
    
    
    

# -*- coding: utf-8 -*-
"""
@author: Stefan Miletic
"""

import cupy as cp
import unittest
import ann
import qn

class TestStringMethods(unittest.TestCase):

    def test_ann(self):
        '''
        Tests if the network constructs the output correctly.
        
        Single input/output and 3 neurons in the hidden layer activated
        with tanh.

        Returns
        -------
        None.

        '''
        small_ann = ann.ffnn(lay_sizes=[3, 1], act_funs=["sigmoid", "id"])
        y = small_ann.forward(x=[3], w=cp.ones(10))
        self.assertEqual(y, 3*cp.tanh(3*1+1)+1)

    def test_qn(self):
        '''
        Tests if the example function f has a minimum at 3.14 using various
        quasi-Newton methods.
        
        f = (x-2)^2 + y^2 + 3.14
        min(f) = 3.14

        Returns
        -------
        None.

        '''
        f = lambda x: (x[0]-2)**2 + x[1]**2 + 3.14
        Df = lambda x: cp.array([2*(x[0]-2), 2*x[1]])
        qn_opt = qn.Optimizers(tol=1e-3)
        
        x_bfgs = cp.array([2.7, 1.3])
        H_bfgs = cp.eye(2)
        
        x_naq = cp.array([2.7, 1.3])
        H_naq = cp.eye(2)
        p = cp.array([0, 0])
        
        x_lbfgs = cp.array([2.7, 1.3])
        SY_lbfgs = list()
        k_lbfgs = 0
        
        x_slbfgs = cp.array([2.7, 1.3])
        
        for i in range(25):
            try:
                x_bfgs, H_bfgs = qn_opt.bfgs(x=x_bfgs,
                                             f=f,
                                             Df=Df,
                                             H=H_bfgs,
                                             c_armijo=1e-3,
                                             rho_armijo=1e-3)
            except:
                pass
            
            try:
                x_naq, p, H_naq = qn_opt.naq(x=x_naq,
                                             f=f,
                                             Df=Df,
                                             p=p,
                                             H=H_naq,
                                             mu=0.8,
                                             c_armijo=1e-3,
                                             rho_armijo=1e-3)
            except:
                pass
            
            try:
                x_lbfgs, SY_lbfgs, k_lbfgs = qn_opt.lbfgs(x=x_lbfgs,
                                                          f=f,
                                                          Df=Df,
                                                          SY=SY_lbfgs,
                                                          m=3,
                                                          k=k_lbfgs,
                                                          c_armijo=1e-3,
                                                          rho_armijo=1e-3)
            except:
                pass
            
            try:
                x_slbfgs = qn_opt.slbfgs(x=x_slbfgs,
                                         f=f,
                                         Df=Df,
                                         m=3,
                                         r=0.1,
                                         c_armijo=1e-3,
                                         rho_armijo=1e-3)
            except:
                pass
            
        bfgs_min = cp.round(f(x_bfgs), 2)
        naq_min = cp.round(f(x_naq), 2)
        lbfgs_min = cp.round(f(x_lbfgs), 2)
        slbfgs_min = cp.round(f(x_slbfgs), 2)
            
        self.assertEqual(bfgs_min, 3.14)
        self.assertEqual(naq_min, 3.14)
        self.assertEqual(lbfgs_min, 3.14)
        self.assertEqual(slbfgs_min, 3.14)

    # def test_split(self):
        

if __name__ == '__main__':
    unittest.main()
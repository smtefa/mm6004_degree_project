# mm6004_degree_project

Code for the bachelor's degree project in mathematics.

* There are several files in this project and each one of them is responsible for certain tasks.
  * File "ann.py" handles creation and execution of neural networks used in this project, as well as their training/testing.
  * File "qn.py" is used for implementation of quasi-Newton methods.
  * File "utils.py" is used for two utility functions necesarry for the technical part in the implementation of quasi-Newton methods.
    It includes the two-loop recursion function used for BFGS and also a function for sampling curvature pairs for SL-BFGS.
  * File "tests.py" is used to test the majority of important functions used in this project. 

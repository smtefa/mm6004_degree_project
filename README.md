# mm6004_degree_project

Code for the bachelor's degree project in mathematics.

* Requirements:
  * Python 3
  * [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for CuPy library used

* There are several files in this project and each one of them is responsible for certain tasks:
  * File [code/ann.py](https://github.com/stmi98/mm6004_degree_project/blob/main/code/ann.py) handles
    creation and execution of neural networks used in this project, as well as their training/testing.
    The networks are built from scratch without external libraries.
  * File [code/qn.py](https://github.com/stmi98/mm6004_degree_project/blob/main/code/qn.py) is used
    for implementation of quasi-Newton methods.
  * File [code/utils.py](https://github.com/stmi98/mm6004_degree_project/blob/main/code/utils.py) is
    used for two utility functions necesarry for the technical part in the implementation of
    quasi-Newton methods. It includes the two-loop recursion function used for BFGS and also a function
    for sampling curvature pairs for SL-BFGS.
  * File [code/tests.py](https://github.com/stmi98/mm6004_degree_project/blob/main/code/tests.py) is used
    to test the majority of important functions used in this project. The focus is set on the correctness of
    neural networks in their construction and the correctness of quasi-Newton methods. Run the file to
    execute the tests.
    
* How to recreate the results:
  * In [code/main.py](https://github.com/stmi98/mm6004_degree_project/blob/main/code/main.py), change
    "w = nn1.train(method="BFGS", params=[1e-3, 1e-3], train_set=Tr1, batch_size=400, epochs=10)" accordingly
    as described in the file, then run the file.

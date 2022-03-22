# bean-classifer
Machine Learning model to classify bean types. Implemented MLP Classifier for multi-class classification. The model is trained on various combinations of hyper-parameters and, the observed results are noted for each combination.

# Data

Features
* Area
* Perimeter
* Major axis length
* Minor axis length
* Aspect ratio
* Convex area
* Equivalent diameter
* Extent
* Solidity
* Roundness
* Compactness
* Shapefactor 1
* Shapefactor 2
* Shapefactor 3
* Shapefactor 4

Classes
* Seker
* Barbunya
* Bombay
* Cali
* Dermason
* Horoz
* Sira


# Hyper-parameters
* Activation function [tanh, relu, Logistic]
* Iterations [50, 100, 150, 200, 250, 300]
* No of hidden layer [[1], [2.3], [7,7,7]]
* Learning rate [0.001, 0.01, 0.1]

# Observations

* Accuracy vs Iterations
![acc_vs_itrs](https://user-images.githubusercontent.com/31441215/159585907-4e6a81fa-1dea-43a8-b0a4-934337b6e633.png)

* Loss vs Iterations
![loss_vs_itrs](https://user-images.githubusercontent.com/31441215/159585932-9fb482ec-817d-40cd-a085-42c32b5aab5b.png)

* R2 score vs Iterations
![r2_vs_itrs](https://user-images.githubusercontent.com/31441215/159585939-fdddbff2-f07c-4f18-aa16-503285fc0a60.png)

* RMSE vs Iterations
![rmse_vs_itrs](https://user-images.githubusercontent.com/31441215/159585944-5a43b822-4503-4ab2-ac20-504b2115be40.png)

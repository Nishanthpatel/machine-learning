# The Effect of Activation Function Choice on Gradient Flow and Trainability in Deep Neural Networks

## Overview
This repository contains the code, figures, and supporting material for the tutorial titled  
“The Effect of Activation Function Choice on Gradient Flow and Trainability in Deep Neural Networks.”

The project investigates how different activation functions—Sigmoid, Tanh, ReLU, and LeakyReLU—affect optimisation behaviour in a deep multilayer perceptron (MLP). A fixed deep architecture is trained on the MNIST dataset under each activation function, and the models are compared using:

- Training loss curves
- Validation accuracy curves
- Layer-wise gradient norm trajectories

The goal is to demonstrate how activation function choice influences gradient flow, convergence speed, and final performance in deep neural networks.

## Repository Structure
The repository is organised as follows:

├── notebook/  
│   └── nishanth.mlcode.ipynb    # Full experiment code and plots  
├── figures/  
│   ├── loss_curves.png                           # Training loss vs epoch  
│   ├── accuracy_curves.png                       # Validation accuracy vs epoch  
│   ├── gradient_norms_sigmoid.png                # Gradient norms for Sigmoid  
│   ├── gradient_norms_tanh.png                   # Gradient norms for Tanh  
│   ├── gradient_norms_relu.png                   # Gradient norms for ReLU  
│   └── gradient_norms_leakyrelu.png              # Gradient norms for LeakyReLU  
├── README.md                                     # This documentation file  
└── LICENSE                                       # Project license (e.g. MIT)  

## Experiment Description

### Architecture
All experiments use the same deep MLP architecture:
- 8 fully connected hidden layers
- 256 neurons per hidden layer
- 10-unit output layer with softmax for MNIST digit classification

The only component that changes between runs is the activation function used in the hidden layers.

### Activation Functions Compared
The following activation functions are evaluated:
- Sigmoid
- Tanh
- ReLU
- LeakyReLU

These functions represent classical saturating activations (Sigmoid, Tanh) and modern non-saturating, piecewise-linear activations (ReLU, LeakyReLU).

### Dataset
The MNIST dataset is used in all experiments:
- 60,000 training images and 10,000 test images
- Grayscale handwritten digits, 28×28 pixels
- Images are normalised and flattened into 784-dimensional vectors

## How to Run the Notebook

1. Clone this repository:
   git clone https://github.com/yourusername/activation-functions-gradient-flow.git  
   cd activation-functions-gradient-flow

2. (Optional) Create and activate a virtual environment.

3. Install the required packages:
   pip install -r requirements.txt

4. Launch Jupyter Notebook:
   jupyter notebook notebook/activation_functions_experiments.ipynb

5. Run all cells in the notebook to:
   - Train the deep MLP with each activation function
   - Generate training loss and validation accuracy plots
   - Compute and plot gradient norm trajectories
   - Save figures into the figures/ directory

## Key Findings (Summary)
The experiments show that:
- Sigmoid suffers from strong vanishing gradients, leading to slow convergence and lower final accuracy.
- Tanh performs better than Sigmoid but still exhibits saturation effects that limit its effectiveness in deeper networks.
- ReLU preserves stronger gradient flow in its active region and converges faster, achieving high validation accuracy.
- LeakyReLU further improves gradient stability by maintaining non-zero gradients for negative inputs, yielding slightly more consistent optimisation behaviour.

These observations support the theoretical understanding that non-saturating, piecewise-linear activation functions are better suited for training deep neural architectures.

## Relation to Tutorial Report
The accompanying report uses the experimental results from this repository to:
- Explain the mathematical properties of each activation function
- Analyse gradient flow using gradient norms
- Compare convergence behaviour and generalisation performance
- Provide practical recommendations for activation function selection in deep networks

## Citation
If you use this repository in academic or instructional work, please cite it as:

[Baddam Nishanth] (2025). “The Effect of Activation Function Choice on Gradient Flow and Trainability in Deep Neural Networks.”  
[Baddam Nishanth] and the repository URL with your actual details.

## License
This project is released under the MIT License (or another licence specified in the LICENSE file). You are free to use, modify, and distribute the code for research and educational purposes, subject to the licence terms.

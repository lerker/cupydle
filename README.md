# cupydle [![Build Status](https://travis-ci.org/lerker/cupydle.svg?branch=master)](https://travis-ci.org/lerker/cupydle) [![GitHub issues](https://img.shields.io/github/issues/lerker/cupydle.svg?style=plastic)](https://github.com/lerker/cupydle/issues) [![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg?style=plastic)](https://raw.githubusercontent.com/lerker/cupydle/master/LICENSE) [![GitHub forks](https://img.shields.io/github/forks/lerker/cupydle.svg?style=plastic)](https://github.com/lerker/cupydle/network) [![todofy badge](https://todofy.org/b/lerker/cupydle)](https://todofy.org/r/lerker/cupydle)

**CU**da**PY**thon**D**eep**LE**arning Neural Networks

:white_check_mark: finished

:negative_squared_cross_mark: not done

:interrobang: in progress or not finished

:bangbang: very important and not done



Functionality:

- Restricted Boltzmann Machine Training
  - [x] This is a complete item With n-step Contrastive Divergence
  - [ ] With persistent Contrastive Divergence
  - [ ] Weight decay, momentum, batch-learning
  - [ ] Binary or gaussian visible nodes

- Restricted Boltzmann Machine Evaluation
  - Sampling from the model
  - Visualizing Filters
  - Annealed Importance Sampling for approximating the partition function
  - Calculating the partition function exactly
  - Visualization and saving of hidden representations

- Stacking RBMs to Deep Belief Networks
  - :negative_squared_cross_mark: Sampling from DBNs


- Neural Network Traing
  - Backpropagation of error
  - RPROP
  - Weight decay, momentum, batch-learning
  - Variable number of layers
  - Cross entropy training

- Finetuning
  - Initalizing a Neural Network with an RBM
  - All of the above functionality can be used

- Training on Image Data
  - Visualization of input, filters and samples from the model
  - on-the-fly modifications to trainingset via gaussian noise or translations

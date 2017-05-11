#!/usr/bin/env python
# =============================================================================
# File Name:     test.py
# Author:        Evan Pete Walsh
# Contact:       epwalsh10@gmail.com
# Creation Date: 2017-05-10
# Last Modified: 2017-05-11 10:55:17
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from myneuralnets import mnist_loader
from myneuralnets import network

path = './data/mnist.pkl.gz'
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(path)
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

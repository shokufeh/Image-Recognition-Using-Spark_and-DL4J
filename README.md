# Image-Recognition-Using-Spark_and-DL4J


Distributed Image recognition with Convolutional Neural Networks using Spark and Deeplearning4J. CNN models adapted from examples on Deeplearning4J.

This repo contains a maven project with two sub-projects:

1) Local

The local project trains the CNNs using the local computer (either with CPU or GPU if available).

2) Spark

The Spark project requires a pre-existing Spark cluster. The Spark cluster is used to distribute the training of CNNs using data-parallelism.

Deeplearning4J

The project makes use of Deeplearning4J to provide an API for constructing, training and evaluating neural networks.

# Geometry parametrization
This repository is a repository to track the work progress on the project "Uncertainty quantification and surrogate modelling with machine learning methods".

**Notice!** In the images below, the **red datapoints** is the original geometry data, while **white** are generated ones.

## Autoencoder (data without surface registration):
![Generated and original geometries](img/result.png)

## Autoencoder (data with surface registration):
![Generated and original geometries](img/dense_autoenc_result.png)

## Convolutional autoencoder (data with surface registration):
![Generated and original geometries](img/conv_autoencoder_result.png)

## Graph neural network
As for now they performed relatevely bad, because of strange connections between the nodes after registration.\
Original:
![Original geometry](img/original_graph.png)
Our simulation inference after 1000 epochs of training simple convolutional graph neural network:
![Inference](img/inference_graph.png)

In ParaView the result looks the following way: \
![Generated and original geometries (graph convolution network)](img/graph_conv_result.png)
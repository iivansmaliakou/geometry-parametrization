# geometry-parametrization
Uncertainty quantification and surrogate modelling with machine learning methods

In the images below, the **red datapoints** is the original geometry data, while **white ones** are generated.

Autoencoder (data without surface registration):
![Generated and original geometries](result.png)

Autoencoder (data with surface registration):
![Generated and original geometries](dense_autoenc_result.png)

Convolutional autoencoder (data with surface registration):
![Generated and original geometries](conv_autoencoder_result.png)

Graph neural network (as for now they performed relatevely bad, because of strange connections between the nodes after registration). \
Original:
![Original geometry](original_graph.png)
Our simulation inference after 1000 epochs of training simple convolutional graph neural network:
![Inference](inference_graph.png)
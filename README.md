# Multi-label-Classification-using-CNN
MLCNN stands for Multi-Label Convolutional Neural Network. It is a type of neural network designed to handle tasks where each input can belong to multiple categories or labels simultaneously. **This model includes**:

    Convolutional Layers: Two convolutional layers with 32 filters (3x3)and (4x4), ReLU activation, L2 regularization, batch normalization, and 70% dropout.
    Flatten Layer: Converts the 2D outputs of the convolutional layers into a 1D vector.
    Fully Connected Layers:
        First layer: 512 units, ReLU, L2 regularization, batch normalization, 70% dropout.
        Second layer: 218 units, ReLU, L2 regularization, batch normalization, 70% dropout.
        Third layer: 256 units, ReLU, L2 regularization, batch normalization, 50% dropout.
    Output Layer: 64 units with sigmoid activation for classification.

This design leverages convolutional layers for feature extraction and fully connected layers for decision making, using regularization and dropout to prevent overfitting.
The optimizer : sgd.
Loss function : Binary cross-entropy loss

## Note:
the input is a 4D matrix (channel matrix of  wireless communication system), and the output is a 2D matrix selected antenna array.
however, a simple change in the preprocessing part of the dataset allows you to use it on different inputs (like images) for multi label classification.

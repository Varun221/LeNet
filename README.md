# IMPLEMENTATION OF LENET DNN MODEL
The LeNet architecture was first introduced by LeCun et al. in their 1998 paper, Gradient-Based Learning Applied to Document Recognition. As the name of the paper suggests, the authors’ implementation of LeNet was used primarily for OCR and character recognition in documents.
The Architecture of LeNet is defined in the image - 

![alt text](https://github.com/Varun221/LeNet/blob/master/LeNet_Architecture.png)

## Description

#### Convolution layer (C1): 
The LeNet architecture expects an input of (32,32,1) image to convert to (28,28,1). To use the architecture for (28,28,1) images we can pad the images to make them to a size of (32,32,1). The activation function used is 'relu' in the code which can be easily changed to 'sigmoid' or 'tanh' based on you requirements.
                    
#### Sub Sampling Layer (S2):
The paper describes a simplified form of a convolutional layer with (2,2) filter of stride 2, We can implement it using custom Keras layers. To increase readability and the performance of the model we can use the regular convolutional layer with the same parameters. Again the activation functions is user's choice.

#### Convolution layer (C3): 
This convolutional layer, according to the paper applies the filter only in specific locations. To simulate it we can use Dropout Class in Keras.

#### Sub Sampling Layer (S4): 
The connection is similar to S2. 

#### Convolutional Layer (C5):
We use 120 filters of size (5x5), apply activation and then flatten the output.

#### Fully connected layer (F6): 
This is a densely connected layer having 84 Units.

#### OUTPUT : 
This is the last layer having 10 Outputs for 10 Classes. The probabilites are estimated using softmax classifier.

Here's the Model Summary - 
![alt text](https://github.com/Varun221/LeNet/blob/master/Model_summary.png)

## Code
The Complete Architecture of the LeNet model is implemented in the Google Colab ipynb file. \
By clicking on "Open in Colab" opens a new tab and the colaboratory, in which you can experiment with the code and run it yourself.

In case you don't want to train the model, The model folder contains two files - 
#### trained_model.hdf5 - 
This is the model trained using original MNIST Data
#### data_augmented_model.hdf5 - 
This is the model trained using original MNISt Data with Data Augmentation. The distortions applied are Width_shift, height_shift and shear range.

The hdf5 file contains - 
* The model's architecture/config
* The model's weight values (which were learned during training)
* The model's compilation information (if compile()) was called
* The optimizer and its state, if any (this enables you to restart training where you left)

You can implement LeNet architecture on you Personal Desktop as follows - 
* Download the hdf5 file.
* Open your IDE and code - 
```python
from tensorflow import keras
model = keras.models.load_model('path/to/hdf5 file')
```
This will load the complete model on your PC.

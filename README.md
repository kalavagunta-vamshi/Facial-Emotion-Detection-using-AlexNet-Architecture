###  Facial Emotion Recognition System Using AlexNet

## Introduction
This repository contains the implementation of a Facial Emotion Recognition system using the AlexNet architecture. Built with Keras, the system employs a pre-trained model to classify human facial expressions into distinct emotions, aiding in the understanding of interpersonal communication and human-computer interaction scenarios.

## Files in the Repository

- **emotion-classification-AlexNet-using-keras.ipynb**: Jupyter notebook detailing the process of training the AlexNet model with Keras.
- **haarcascade_frontalface_default.xml**: XML configuration for Haar Cascade, used to detect faces in the images.
- **main.py**: The Python script that interfaces with the camera to perform real-time emotion detection using the trained model.
- **model.h5**: The trained AlexNet model saved in H5 format that predicts emotions from facial expressions.

## How to Set Up and Run

### Prerequisites
Ensure you have Python installed along with the following packages: TensorFlow, Keras, and OpenCV. You can install them using pip:
```bash
pip install tensorflow keras opencv-python
```

### Installation
1. Clone this repository to your local machine.
   ```bash
   git clone <https://github.com/kalavagunta-vamshi/Facial-Emotion-Detection-using-AlexNet-Architecture.git>
   ```

2. Navigate to the cloned directory.
   ```bash
   cd <https://github.com/kalavagunta-vamshi/Facial-Emotion-Detection-using-AlexNet-Architecture.git>
   ```

### Execution
Run the following command to start the real-time emotion recognition:
```bash
python main.py
```

## Model Description: AlexNet Architecture

AlexNet is a pioneering convolutional neural network (CNN) that significantly advanced the field of deep learning by winning the ImageNet Large Scale Visual Recognition Challenge in 2012. Here’s a brief on the architecture:

- **Input Layer**: Accepts an image of size 227x227 pixels as input.
- **Convolutional Layers**: Consists of five convolutional layers; some use max pooling and all are followed by ReLU activation functions to introduce non-linearities.
- **Fully Connected Layers**: Follows the convolutional layers with three fully connected layers, the last of which leads to a 1000-way softmax output for ImageNet but is adjusted according to the number of emotions classified in this project.
- **Output Layer**: Outputs the probability distribution across the classified emotions.

### Specific Modifications for Emotion Recognition:
For this project, the output layer dimensions and some internal structuring may have been modified to better suit the nuances of facial emotion recognition.





import numpy as np
import random 
from PIL import Image
import matplotlib.pyplot as plt
import os

DATASIZE = 500
Fsize = DATASIZE // 2
Zsize = DATASIZE - Fsize
XTRAIN = [ ] 
YTRAIN = [ ]

data_set_dir = "Data/"

# random seleting data is not good for a neural network
for i in range(Fsize):
    ran_img = random.choice(["1.png", "2.png", "3.png", "4.png", "5.png"]) 
    file_path = os.path.join(data_set_dir, "0", ran_img)
    img = Image.open(file_path).convert("L")     #convert RGBA to gray
    img_arr = np.array(img)
    img_arr = img_arr / 255.0
    XTRAIN.append(img_arr)
    YTRAIN.append(0)

for i in range(Zsize):
    ran_img = random.choice(["1.png", "2.png", "3.png", "4.png", "5.png"]) 
    file_path = os.path.join(data_set_dir, "1", ran_img)
    img = Image.open(file_path).convert("L")     #convert RGBA to gray
    img_arr = np.array(img)
    img_arr = img_arr / 255.0
    XTRAIN.append(img_arr)
    YTRAIN.append(1)

class NeuralNetwork:
    def __init__( self, _inputSize, _outputSize): 
        self.inputSize = _inputSize
        self.outputSize = _outputSize
        self.initWeights()

    def initWeights(self):
        self.l1_weights = np.random.randn(10, self.inputSize) # 10 x 784
        self.l1_bias = np.random.randn(10)

        self.l2_weights = np.random.randn(5, 10) 
        self.l2_bias = np.random.randn(5)

        self.l3_weights = np.random.randn(self.outputSize, 5) # 1 x 5
        self.l3_bias = np.random.randn(self.outputSize)

    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * ( 1 - x )

        return 1 / (1 + np.exp(-x))

    def feedForward(self, inputs):
        l1_output = self.sigmoid(np.dot(self.l1_weights, inputs)    + self.l1_bias)
        l2_output = self.sigmoid(np.dot(self.l2_weights, l1_output) + self.l2_bias)
        l3_output = self.sigmoid(np.dot(self.l3_weights, l2_output) + self.l3_bias)

        return l1_output, l2_output, l3_output

    def binary_cross_entropy(self, predicted, actual, deriv=False):
        if deriv:
            epsilon = 1e-15
            predicted = np.clip(predicted, epsilon, 1 - epsilon)
            return (predicted - actual) / (predicted * (1 - predicted))
        
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return - (actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    def backpropagation(self, inputs, l1_out, l2_out, l3_out, actual, eta):
        delta_l3 = self.binary_cross_entropy(l3_out[0], actual, deriv=True) * self.sigmoid(l3_out, deriv=True)
        delta_l2 = np.dot(self.l3_weights.T, delta_l3) * self.sigmoid(l2_out, deriv=True) # 1 x 5 * 1, 5 x 1 * 1
        delta_l1 = np.dot(self.l2_weights.T, delta_l2) * self.sigmoid(l1_out, deriv=True)


        self.l3_weights -= eta * np.outer(delta_l3, l2_out)
        self.l3_bias -= eta * delta_l3

        self.l2_weights -= eta * np.outer(delta_l2, l1_out)
        self.l2_bias -= eta * delta_l2

        self.l1_weights -= eta * np.outer(delta_l1, inputs)
        self.l1_bias -= eta * delta_l1

    def train( self, xset, yset, epochs, eta):
        for epoch in range(epochs):
            error = 0
            for itr in range(DATASIZE):
                inputs = xset[itr]
                desired = yset[itr]

                l1_out, l2_out, l3_out = self.feedForward(inputs.flatten())
                error_raw = l3_out[0] - desired
                error += self.binary_cross_entropy(l3_out[0], desired)
                self.backpropagation(inputs, l1_out, l2_out, l3_out, desired, eta)

            print(f"Epoch : {epoch + 1} Error : {error / DATASIZE}")

            array = self.l1_weights[0]
            img_array = array.reshape((28,28))
            img = Image.fromarray(img_array.astype("uint8"))
            img.save(f"output{epoch}.jpg")


discriminator = NeuralNetwork(28 * 28, 1)

discriminator.train(XTRAIN, YTRAIN, 1000, 0.01)

# while True:
#     file_name = input("File name : ")
#     img = Image.open(file_name).convert("L")
#     img_arr = np.array(img)
#     img_arr = img_arr / 255.0
#     _, _, output = discriminator.feedForward(img_arr.flatten())
#     print(f"Model thinks it is a : raw - {output}, rounded - {round(output[0])}")

















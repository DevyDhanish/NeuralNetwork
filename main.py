import numpy as np
import random 
from PIL import Image
import os

DATASIZE = 100
XTRAIN = [ ] 
YTRAIN = [ ]

data_set_dir = "Data/"

for i in range(DATASIZE):
    ran_ch = random.choice(["0", "1"])
    ran_img = random.choice(["1.png", "2.png", "3.png", "4.png", "5.png"]) 
    file_path = os.path.join(data_set_dir, ran_ch, ran_img)
    img = Image.open(file_path).convert("L")     #convert RGBA to gray
    img_arr = np.array(img)
    img_arr = img_arr / 255.0
    XTRAIN.append(img_arr)
    YTRAIN.append(int(ran_ch))

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

        self.l3_weights = np.random.randn(self.outputSize, 5)
        self.l3_bias = np.random.randn(1)

    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * ( 1 - x )

        return 1 / (1 + np.exp(-x))

    def feedForward(self, inputs):
        l1_output = self.sigmoid(np.dot(self.l1_weights, inputs)    + self.l1_bias)
        l2_output = self.sigmoid(np.dot(self.l2_weights, l1_output) + self.l2_bias)
        l3_output = self.sigmoid(np.dot(self.l3_weights, l2_output) + self.l3_bias)

        return l1_output, l2_output, l3_output

    def train( self, epochs, eta):
        for epoch in range(epochs):
            error = 0
            for itr in range(DATASIZE):
                inputs = XTRAIN[itr]
                desired = YTRAIN[itr]

                l1_out, l2_out, l3_out = self.feedForward(inputs.flatten())

                break
            break

model = NeuralNetwork(28 * 28, 1)

model.train(1, 0.01)

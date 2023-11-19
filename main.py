import random
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def derived_sigmoid(x):
    f = sigmoid(x)
    return f * (1 - f)

def dot(x: list, y: list):
    scalar = 0
    for index, _ in enumerate(x):
        scalar += x[index] * y[index]

    return scalar

def sum(x: list):
    sum = 0
    for value in x:
        sum += value

    return sum

def mse(y: list, yPred: list):
    value = 0
    for index, _ in enumerate(y):
        value += (y[index] - yPred[index])**2

    return value/len(y)

class Neuron:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs) -> float:
        value = dot(inputs, self.weights) + self.bias
        f_sigmoid = sigmoid(value)
        print(f'Feed forward. S: {f_sigmoid} V: {value}')
        return f_sigmoid
    
    def extendedFeedForward(self, inputs) -> tuple:
        value = dot(inputs, self.weights) + self.bias
        f_sigmoid = sigmoid(value)
        print(f'Extended feed forward. S: {f_sigmoid} V: {value}')
        return (f_sigmoid, value)

'''
params: [
    [weights, bias]
]

[X1] -- [H1]
     \/      \ |OUTPUT|
     /\      / |      |
[X2] -- [H2]
'''
class NeuralNetwork:
    def __init__(self, inputs, output) -> None:
        self.output = Neuron(output[0], output[1])
        self.minmaxScalingParams = []
        self.networks = []
        for input in inputs:
            self.networks.append(Neuron(input[0], input[1])) # weights, bias

    def getMinmaxParams(self, column):
        colMin = colMax = column[0]
        for col_item in column:
            if colMin > col_item:
                colMin = col_item
            if colMax < col_item:
                colMax = col_item

        return (colMin, colMax)

    def minmaxCalc(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin)

    def minmaxScaling(self, data):
        if len(self.minmaxScalingParams) > 0:
            scaled = [
                self.minmaxCalc(data[0], self.minmaxScalingParams[0][0], self.minmaxScalingParams[0][1]),
                self.minmaxCalc(data[1], self.minmaxScalingParams[1][0], self.minmaxScalingParams[1][1])
            ]
            return scaled
        
        col1 = []
        col2 = []
        for data_item in data:
            col1.append(data_item[0])
            col2.append(data_item[1])

        col1MinmaxParams = self.getMinmaxParams(col1)
        col2MinmaxParams = self.getMinmaxParams(col2)
        self.minmaxScalingParams.append([col1MinmaxParams[0], col1MinmaxParams[1]])
        self.minmaxScalingParams.append([col2MinmaxParams[0], col2MinmaxParams[1]])

        for index, _ in enumerate(data):
            data[index][0] = self.minmaxCalc(data[index][0], col1MinmaxParams[0], col1MinmaxParams[1])
            data[index][1] = self.minmaxCalc(data[index][1], col2MinmaxParams[0], col2MinmaxParams[1])

        return data

    def feedForward(self, inputs) -> float:
        feedForwardList = []
        for neural in self.networks:
            feedForwardList.append(neural.feedForward(inputs))

        return self.output.feedForward(feedForwardList)
    
    def extendedFeedForward(self, inputs) -> tuple:
        feedForwardList = []
        for neuron in self.networks:
            feedForwardList.append(neuron.extendedFeedForward(inputs))

        return self.output.extendedFeedForward(feedForwardList)
    
    def train(self, data, answers, showLossPlot):
        rate = 0.1
        epochs = 1000
        plotDataX = []
        plotDataY = []
        for epoch in range(epochs):
            for x, y_entity in zip(data, answers):
                h_pred_list = []
                for neuron in self.networks:
                    h_pred_list.append(neuron.extendedFeedForward(x))

                h_pred_out = []
                for h_pred in h_pred_list:
                    h_pred_out.append(h_pred[0])

                y_pred = self.output.extendedFeedForward(h_pred_out)
                print(f'Predicted: {y_pred[0]}')

                dl_dy_pred = -2*(y_entity - y_pred[0])
                dy_pred_dw = []
                for _, h_pred in h_pred_list:
                    dy_pred_dw.append(h_pred * derived_sigmoid(y_pred[1]))

                dy_pred_dh = []
                for weight in self.output.weights:
                    dy_pred_dh.append(weight * derived_sigmoid(y_pred[1]))
                
                dy_pred_db = []
                allNeuronList = self.networks + [self.output]
                for neuron in allNeuronList:
                    sum = neuron.extendedFeedForward(x)[1]
                    print(f'Neuron weights: {neuron.weights}')
                    dy_pred_db.append(derived_sigmoid(sum))

                d_hn_dw = []
                for neuron in self.networks:
                    for x_entity in x:
                        sum = neuron.extendedFeedForward(x)[1]
                        d_hn_dw.append(x_entity * derived_sigmoid(sum))

                paramCollector = []
                for index, neuron in enumerate(self.networks):
                    param = {
                            "weights": [d_hn_dw[index * 2], d_hn_dw[index * 2 + 1]],
                            "bias": dy_pred_db[index],
                            "dy_pred_dh": dy_pred_dh[index]
                        }
                    paramCollector.append(param) # pair

                print(f'h predicated: {h_pred_list}')
                print(f'dy pred dh: {dy_pred_dh}')
                print(f'dy pred db: {dy_pred_db}')
                print(f'dy hn dw: {d_hn_dw}')
                print(f'rebalancing param collector: {paramCollector}')

                # Updating params
                # Updating hidden layers
                for nIndex, neuron in enumerate(self.networks):
                    for wIndex, _ in enumerate(neuron.weights):
                        print(f'Old weights: {neuron.weights}')
                        neuron.weights[wIndex] -= rate * dl_dy_pred * paramCollector[nIndex]["dy_pred_dh"] * paramCollector[nIndex]["weights"][wIndex]
                        print(f'New weights: {neuron.weights}')
                    
                    print(f'Old bias: {neuron.bias}')
                    neuron.bias -= rate * dl_dy_pred * paramCollector[nIndex]["dy_pred_dh"] * paramCollector[nIndex]["bias"]
                    print(f'New bias: {neuron.bias}')

                # Updating output layer
                for wIndex, _ in enumerate(self.output.weights):
                    print(f'Old output weights: {self.output.weights}')
                    self.output.weights[wIndex] -= rate * dl_dy_pred * dy_pred_dw[wIndex]
                    print(f'New output weights: {self.output.weights}')

                print(f'Old output bias: {self.output.bias}')
                self.output.bias -= rate * dl_dy_pred * dy_pred_dw[wIndex] * dy_pred_db[len(dy_pred_db) - 1]
                print(f'New output bias: {self.output.bias}')

            if epoch % 10 == 0:
                y_preds = []
                for data_item in data:
                    y_preds.append(self.feedForward(data_item))

                loss = mse(answers, y_preds)
                if showLossPlot:
                    plotDataX.append(epoch)
                    plotDataY.append(loss)
                print(f'Epoch {epoch} loss = {loss}')
        
        if showLossPlot:
            plt.plot(plotDataX, plotDataY)
            plt.title("Loss")
            plt.show()


nn = NeuralNetwork([
        [[random.gauss(0, 1.0), random.gauss(0, 1.0)], 0],
        [[random.gauss(0, 1.0), random.gauss(0, 1.0)], 0]],
        [[random.gauss(0, 1.0), random.gauss(0, 1.0)], 0]
    )

data = [
        [50, 160],
        [75, 172],
        [90, 180],
        [60, 170],
        [55, 165],
        [78, 174],
        [101, 182],
        [63, 169],
    ]

answers = [1, 0, 0, 1, 1, 0, 0, 1]

nn.minmaxScaling(data)

nn.train(data, answers, False)

emily = [52, 160]
frank = [63, 173]
alice = [85, 176]

emily = nn.minmaxScaling(emily)
frank = nn.minmaxScaling(frank)
alice = nn.minmaxScaling(alice)

print(f'Emily: {emily} -> {nn.feedForward(emily)}')
print(f'Frank: {frank} -> {nn.feedForward(frank)}')
print(f'Alice: {alice} -> {nn.feedForward(alice)}')
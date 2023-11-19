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

    def feed_forward(self, inputs) -> float:
        value = dot(inputs, self.weights) + self.bias
        f_sigmoid = sigmoid(value)
        print(f'Feed forward. S: {f_sigmoid} V: {value}')
        return f_sigmoid
    
    def extended_feed_forward(self, inputs) -> tuple:
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
        self.minmax_scaling_params = []
        self.networks = []
        for input in inputs:
            self.networks.append(Neuron(input[0], input[1])) # weights, bias

    def get_minmax_params(self, column):
        colMin = colMax = column[0]
        for col_item in column:
            colMin = min(colMin, col_item)
            colMax = max(colMax, col_item)

        return (colMin, colMax)

    def minmax_calc(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin)
    
    def inverse_minmax_calc(self, x_scaled, xmin, xmax):
        return x_scaled * (xmax - xmin) + xmin
    
    def inverse_minimax_scaling(self, data):
        if len(self.minmax_scaling_params) > 0:
            inversed = [
                self.inverse_minmax_calc(data, self.minmax_scaling_params[2][0], self.minmax_scaling_params[2][1])
            ]
            return inversed

    def minmax_scaling(self, data):
        if len(self.minmax_scaling_params) > 0:
            scaled = []
            for index, _ in enumerate(data):
                scaled.append(self.minmax_calc(data[index], self.minmax_scaling_params[index][0], self.minmax_scaling_params[index][1]))

            return scaled
        
        for colIndex, _ in enumerate(data[0]):
            col = []
            for rowIndex, _ in enumerate(data):
                col.append(data[rowIndex][colIndex])
            
            col_minmax_parms = self.get_minmax_params(col)
            self.minmax_scaling_params.append([col_minmax_parms[0], col_minmax_parms[1]])

        for dIndex, data_item in enumerate(data):
            for item_index, _ in enumerate(data_item):
                data[dIndex][item_index] = self.minmax_calc(
                    data[dIndex][item_index], 
                    self.minmax_scaling_params[item_index][0], 
                    self.minmax_scaling_params[item_index][1]
                )

        return data

    def feed_forward(self, inputs) -> float:
        feed_forward_list = []
        for neural in self.networks:
            feed_forward_list.append(neural.feed_forward(inputs))

        return self.output.feed_forward(feed_forward_list)
    
    def extended_feed_forward(self, inputs) -> tuple:
        feed_forward_list = []
        for neuron in self.networks:
            feed_forward_list.append(neuron.extended_feed_forward(inputs))

        return self.output.extended_feed_forward(feed_forward_list)
    
    def train(self, data, answers, epochs, rate, showLossPlot):
        plot_data_x = []
        plot_data_y = []
        for epoch in range(epochs):
            for x, y_entity in zip(data, answers):
                h_pred_list = []
                for neuron in self.networks:
                    h_pred_list.append(neuron.extended_feed_forward(x))

                h_pred_out = []
                for h_pred in h_pred_list:
                    h_pred_out.append(h_pred[0])

                y_pred = self.output.extended_feed_forward(h_pred_out)
                print(f'Predicted: {y_pred[0]}')

                dl_dy_pred = -2*(y_entity - y_pred[0])
                dy_pred_dw = []
                for _, h_pred in h_pred_list:
                    dy_pred_dw.append(h_pred * derived_sigmoid(y_pred[1]))

                dy_pred_dh = []
                for weight in self.output.weights:
                    dy_pred_dh.append(weight * derived_sigmoid(y_pred[1]))
                
                dy_pred_db = []
                all_neuron_list = self.networks + [self.output]
                for neuron in all_neuron_list:
                    sum = neuron.extended_feed_forward(x)[1]
                    print(f'Neuron weights: {neuron.weights}')
                    dy_pred_db.append(derived_sigmoid(sum))

                d_hn_dw = []
                for neuron in self.networks:
                    for x_entity in x:
                        sum = neuron.extended_feed_forward(x)[1]
                        d_hn_dw.append(x_entity * derived_sigmoid(sum))

                param_collector = []
                for index, neuron in enumerate(self.networks):
                    param = {
                            "weights": [d_hn_dw[index * 2], d_hn_dw[index * 2 + 1]],
                            "bias": dy_pred_db[index],
                            "dy_pred_dh": dy_pred_dh[index]
                        }
                    param_collector.append(param) # pair

                print(f'h predicated: {h_pred_list}')
                print(f'dy pred dh: {dy_pred_dh}')
                print(f'dy pred db: {dy_pred_db}')
                print(f'dy hn dw: {d_hn_dw}')
                print(f'rebalancing param collector: {param_collector}')

                # Updating params
                # Updating hidden layers
                for nIndex, neuron in enumerate(self.networks):
                    for wIndex, _ in enumerate(neuron.weights):
                        print(f'Old weights: {neuron.weights}')
                        neuron.weights[wIndex] -= rate * dl_dy_pred * param_collector[nIndex]["dy_pred_dh"] * param_collector[nIndex]["weights"][wIndex]
                        print(f'New weights: {neuron.weights}')
                    
                    print(f'Old bias: {neuron.bias}')
                    neuron.bias -= rate * dl_dy_pred * param_collector[nIndex]["dy_pred_dh"] * param_collector[nIndex]["bias"]
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
                    y_preds.append(self.feed_forward(data_item))

                loss = mse(answers, y_preds)
                plot_data_x.append(epoch)
                plot_data_y.append(loss)
                print(f'Epoch {epoch} loss = {loss}')

            print(f'Last epoch: loss = {plot_data_y[-1]}')

        if showLossPlot:
            plt.plot(plot_data_x, plot_data_y)
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
        [63, 169]
    ]

answers = [1, 0, 0, 1, 1, 0, 0, 1]

nn.minmax_scaling(data)

nn.train(data, answers, 1000, 0.1, True)

emily = [52, 160]
frank = [63, 173]
dan = [93, 178]
alice = [85, 176]
eu = [59, 162]

emily = nn.minmax_scaling(emily)
frank = nn.minmax_scaling(frank)
alice = nn.minmax_scaling(alice)
eu = nn.minmax_scaling(eu)
dan = nn.minmax_scaling(dan)

print(f'Emily: {emily} -> {nn.feed_forward(emily)}')
print(f'Frank: {frank} -> {nn.feed_forward(frank)}')
print(f'Alice: {alice} -> {nn.feed_forward(alice)}')
print(f'Eugenia: {eu} -> {nn.feed_forward(eu)}')
print(f'Dan: {dan} -> {nn.feed_forward(dan)}')
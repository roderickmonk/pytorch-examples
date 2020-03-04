from __future__ import print_function
from random import choices
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):

        outputs = []

        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            # print ("input_t: ", type(input_t), input_t.shape)

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            # print ("h_t2: ", type(h_t2), h_t2.shape)

            output = self.linear(h_t2)
            # print ("output: ", type(output), output.shape)

            outputs += [output]

        # print ("outputs[0] (1): ", outputs[0])

        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        # print ("outputs[0] (2): ", outputs[0])

        outputs = torch.stack(outputs, 1).squeeze(2)

        print("outputs.shape: ", outputs.shape)

        return outputs


if __name__ == '__main__':

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    print("data\n", data)
    print("data.shape: ", data.shape)

    input = torch.from_numpy(data[3:, :-1])
    print("input\n", input)
    print("type (input): ", type(input))
    print("input.shape: ", input.shape)

    noise = 0.0
    # noise = np.random.normal(0, 0.01, input.size(1))

    target = torch.from_numpy(data[3:, 1:])
    print("target.shape: ", target.shape)

    test_input = torch.from_numpy(data[:3, :-1] + noise)
    print("test_input.shape: ", test_input.shape)

    test_target = torch.from_numpy(data[:3, 1:])
    print("test_target.shape: ", test_target.shape)

    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # begin to train
    for i in range(15):

        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000

            pred = seq(test_input[:1], future=future)

            print("pred: ", pred)
            print("pred.shape: ", pred.shape)

            loss = criterion(pred[:, :-future], test_target[:1])
            print('test loss:', loss.item())
            y = pred.detach().numpy()
            print("y: ", y)
            print("y.shape: ", y.shape)

        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title(
            'Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)),
                     yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future),
                     yi[input.size(1):], color + ':', linewidth=2.0)
        draw(y[0], 'r')
        #draw(y[1], 'g')
        #draw(y[2], 'b')
        plt.savefig('predict%d.pdf' % i)
        plt.close()

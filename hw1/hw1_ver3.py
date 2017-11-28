import numpy as np
import csv
import pandas as pd


def read_csv(data_size): #read csv files
    # np.set_printoptions(threshold = np.nan)
    data = np.genfromtxt("train.csv", delimiter=",")
    output = []
    data_final = []
    data = np.delete(data, np.s_[0:3], 1)
    data = np.transpose(np.delete(data, 0, 0))
    for i in range(0, data.shape[0]):
        output.append(np.split(data[i], data.shape[1] / data_size))

    for i in range(0, len(output[0])):
        for j in range(0, data.shape[0]):
            data_final.append(output[j][i])
    data_final = np.array(data_final, dtype=float)
    data_final[np.isnan(data_final)] = 0
    return data_final


def gradient_run(data_train, days, weights, N, learning_rate):
    gradients = np.zeros(163, dtype=float)
    error = 0.0
    error_squared_sum = 0.0
    # prepare training data
    for i in range(0, N):
        data_train_slot = data_train[i:i + days]
        y_target = data_train[i + days][9]

        # for j in range(0,days):
        features = np.array(np.concatenate(np.split(data_train_slot,9), axis=1))
        features = np.insert(features,0,1)
        # cost function
        error_squared_sum += (y_target - np.dot(features, weights)) ** 2

        # gradient descent parameter
        gradients += -2 * (y_target - np.dot(features, weights)) * features

    weights = weights - learning_rate * gradients
    return weights, error_squared_sum


def error_cal(features, weights, y_target, N):
    # cost function
    error = ((y_target - np.dot(features, weights)) ** 2) / N
    return error


def main():
    # parameters
    days = 9
    data_size = 18
    learning_rate = 0.0000000001
    iter_num = 200000

    # read training data
    data_train = read_csv(data_size)

    N = len(data_train) - 10
    # training
    weights = np.random.rand(days * data_size + 1)
    # weights = np.genfromtxt("weights.csv", delimiter=",")
    print("initial weight" + str(weights[0]))
    error_old = 0.0
    for j in range(0, iter_num):
        print(j)
        # run gradient descent
        weights, error = gradient_run(data_train, days, weights, N, learning_rate)

        print("error_sum=" + str(error) + "weights = " + str(weights[0]) + 'gaps = ' + str(abs(error - error_old)))
        if abs(error - error_old) <= 0.00001:
            break
        error_old = error
    print('FINAL: weights[0] = ' + str(weights[0]) + ', error = ' + str(error) )
    np.savetxt('weights.csv', weights,delimiter =',')

if __name__ == '__main__':
    main()

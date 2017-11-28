import numpy as np
import csv
import pandas as pd



def read_csv(data_size): # read training data

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


###### i think my problem is here!!!
def gradient_run(data_train, days, weights, N, learning_rate):
    gradients = np.zeros(163, dtype=float)
    error = 0.0
    error_squared_sum = 0.0
    # prepare training data
    for i in range(0, N):
        data_train_slot = data_train[i:i + days]
        y_target = data_train[i + days][9]
        # 1 is the feature for bias
        bias = np.expand_dims(np.array([1]), axis=0)
        features = np.concatenate((bias, np.reshape(data_train_slot, (1, 162))), axis=1)
        error += (y_target-np.dot(features,weights))
        error_squared_sum += (y_target-np.dot(features,weights))**2
    # loss function here


    for i in range(0, features.shape[1]):
        gradients[i] = -2 * error * features[0][i]
    weights = weights - learning_rate * gradients
    print("error_sum=" + str(error_squared_sum[0]) +"error = " +str(error)+ "weights = " + str(weights[0]))
    return weights


def error_cal(features, weights, y_target, N):
    # cost function
    error = ((y_target - np.dot(features, weights)) ** 2) / N
    return error


def main():
    # parameters
    days = 9
    data_size = 18
    learning_rate = 0.0000000002
    iter_num = 40

    # read training data
    data_train = read_csv(data_size)
    N = len(data_train) - 10
    # training
    weights = np.random.rand(days * data_size + 1)
    print("initial weight" + str(weights[0]))
    for j in range(0, iter_num):

        # run gradient descent
        # print(j)
        weights = gradient_run(data_train, days, weights, N, learning_rate)




    print (weights)

if __name__ == '__main__':
    main()

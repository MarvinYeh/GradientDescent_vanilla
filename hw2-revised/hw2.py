import numpy as np
import csv
import pandas as pd
import math
import sys

def read_csv(data_size): #read csv files
    # np.set_printoptions(threshold = np.nan)
    data = np.genfromtxt("train.csv", delimiter=",")
    # data = np.genfromtxt(sys.argv[1],delimiter=",")
    output = []
    data_final = []
    data = np.delete(data, np.s_[0:3], 1)
    data = np.transpose(np.delete(data, 0, 0))

    # convert rain data into 0 and 1
    for j in range(0,int(data.shape[1]/data_size)):
        temp = data[:,j*18+10]
        temp[~np.isnan(temp)]=1
        temp[np.isnan(temp)] = 0
        data[:,j*18+10] = temp

    # prepare data
    for i in range(0, data.shape[0]):
        output.append(np.split(data[i], data.shape[1] / data_size))
    for i in range(0, len(output[0])):
        for j in range(0, data.shape[0]):
            data_final.append(output[j][i])
    data_final = np.array(data_final, dtype=float)

    return data_final

def sigmoid(x):
    sig = 1/(1+math.exp(-x))
    return sig

def gradient_run(data_train, days, weights, N, learning_rate):
    gradients = np.zeros(163, dtype=float)
    cross_entropy = 0.0
    # prepare training data
    for i in range(0, N):
        data_train_slot = data_train[i:i + days]
        y_target = data_train[i + days][10]
        features = np.array(np.concatenate(np.split(data_train_slot,9), axis=1))
        features = np.insert(features, 0,1)
        # cost function
        f_wb = sigmoid(np.dot(features,weights))

        cross_entropy += -(y_target* sigmoid(np.log(f_wb))+(1-y_target)*(np.log(1-f_wb)))

        # gradient descent parameter
        gradients += -2 * (y_target - f_wb) * features

    weights = weights - learning_rate * gradients

    return weights, cross_entropy

def predict(weights):
    data = np.genfromtxt("test_X.csv", delimiter=",")
    data = np.delete(data, np.s_[0:2],1)
    y_predict = np.zeros(240,dtype = float)
    for i in range(0,int(len(data)/18)):
        # features.append(np.split(np.transpose(data[i:i+18]),9))
        features = np.concatenate(np.split(np.transpose(data[i:i+18]),9), axis = 1)
        features = np.insert(features, 0, 1)
        features[np.isnan(features)]=0
        y_predict[i] = np.dot(features,weights)
    return y_predict


def main():
    # parameters
    days = 9
    data_size = 18
    learning_rate = 0.0000000005
    iter_num = 10000
    cross_entropy_old = 0;
    # read training data
    data_train = read_csv(data_size)
    N = len(data_train) - 10
    # training

    #weights = np.ones(days * data_size + 1)/2000
    weights = weights = np.genfromtxt("weights.csv", delimiter=",")

    print("initial weight" + str(weights[0]))
    for j in range(0, iter_num):


        # run gradient descent
        weights,cross_entropy = gradient_run(data_train, days, weights, N, learning_rate)
        gap = abs(cross_entropy-cross_entropy_old)
        print('iter = '+ str(j) + " ,weights = " + str(weights[0]) + ', XEnt = ' + str(cross_entropy)+' ,gaps = '+str(gap))
        if gap <=0.001:
            break
        cross_entropy_old = cross_entropy
    # print("weights = ", + str(weights))
    np.savetxt("hw2_test.txt",weights,delimiter= ',',fmt = '%10.5f')





if __name__ == '__main__':
    main()

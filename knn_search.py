'''
Nearest neighbor search using Sklearn KNN
Author: Lili Meng
Date: August 28th, 2018
'''

import cv2
import numpy as np
import argparse
import os
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser(description='feature extractor from pretrained model')
parser.add_argument('--query_data_path', default='/Users/lilimeng/Desktop/Price_prediction/extract_features/saved_test_features/', type=str, help='test data path')

parser.add_argument('--train_data_path', default='/Users/lilimeng/Desktop/Price_prediction/extract_features/saved_train_features/', type=str, help='train data path')

def knn_search(query_data, training_data):
    knn = NearestNeighbors(2, 0.4)
    knn.fit(training_data)
    results = knn.kneighbors(query_data, 2, return_distance=False)

    return results


def main():

    global arg
    arg = parser.parse_args()
    print(arg)

    query_data = np.load(os.path.join(arg.query_data_path, 'features_00000.npy'))
    query_data_np = np.expand_dims(query_data, axis=0)

    print("query_data.shape: ", query_data.shape)
    print("query_data_np.shape: ", query_data_np.shape)
    all_train_data_list=[]

    for i in range(50):
        single_train_data_path = os.path.join(arg.train_data_path, 'features_{}.npy'.format('%05d'%i))
        single_train_data = np.load(single_train_data_path)
       
        all_train_data_list.append(single_train_data)


    all_train_data_np = np.asarray(all_train_data_list)

    print("all_train_data_np.shape: ", all_train_data_np.shape)
    results = knn_search(query_data_np, all_train_data_np)

    print(results)
if __name__=='__main__':
    main()

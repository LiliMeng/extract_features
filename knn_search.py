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
import scipy.io as sio 


parser = argparse.ArgumentParser(description='feature extractor from pretrained model')
#parser.add_argument('--query_data_path', default='/Users/lilimeng/Desktop/Price_prediction/extract_features/saved_test_features/', type=str, help='test data path')
parser.add_argument('--query_img_name', default='/Users/lilimeng/Desktop/Price_prediction/SpaceNet/spaceNet_annotations_Paris/spaceNet_paris_train_jpg/RGB-PanSharpen__2.21692139996_49.0319109.jpg', type=str, help='query img name')
parser.add_argument('--query_img_list', default='/Users/lilimeng/Desktop/Price_prediction/extract_features/img_list/database_list.txt', type=str, help='query data list')
parser.add_argument('--query_features_path', default='/Users/lilimeng/Desktop/Price_prediction/extract_features/saved_resnet_database_features/', type=str, help='the path for storing query features')
parser.add_argument('--database_featuers_path', default='/Users/lilimeng/Desktop/Price_prediction/extract_features/saved_resnet_database_features/', type=str, help='the path for storing database features')
parser.add_argument('--database_img_path', default='/Users/lilimeng/Desktop/Price_prediction/SpaceNet/spaceNet_annotations_Paris/spaceNet_paris_train_jpg', type=str, help='train data path')
parser.add_argument('--database_img_list', default='/Users/lilimeng/Desktop/Price_prediction/extract_features/img_list/database_list.txt', type=str, help = 'database img list')


def knn_search(query_data, training_data):
    knn = NearestNeighbors(2, 0.4)
    knn.fit(training_data)
    results = knn.kneighbors(query_data, 4, return_distance=False)

    return results


def main():

    global arg
    arg = parser.parse_args()
    print(arg)

    query_img_lines = [line.strip() for line in open(arg.query_img_list).readlines()]

    print("arg.query_img_name: ", arg.query_img_name)
    query_img = cv2.imread(arg.query_img_name)

    #cv2.imshow('query img', query_img)
    #cv2.waitKey(0)

    for i in range(len(query_img_lines)):
        print("query_img_lines[i]: ", query_img_lines[i])
        print("arg.query_img_name.split('/')[-1] ", arg.query_img_name.split('/')[-1])
        if query_img_lines[i] == arg.query_img_name.split('/')[-1]:
            feature_index = i 

    query_feature_name = 'features_{}.npy'.format('%05d'%feature_index)
    print("query feature name: ", query_feature_name)
    
    query_data = np.load(os.path.join(arg.query_features_path, query_feature_name))
    query_data_np = np.expand_dims(query_data, axis=0)

    print("query_data.shape: ", query_data.shape)
    print("query_data_np.shape: ", query_data_np.shape)
    all_train_data_list=[]
    train_database_lines = [line.strip() for line in open(arg.database_img_list).readlines()]

    for i in range(len(train_database_lines)):
        single_train_data_path = os.path.join(arg.database_featuers_path, 'features_{}.npy'.format('%05d'%i))
        single_train_data = np.load(single_train_data_path)
       
        all_train_data_list.append(single_train_data)


    all_train_data_np = np.asarray(all_train_data_list)

    print("all_train_data_np.shape: ", all_train_data_np.shape)
    results = knn_search(query_data_np, all_train_data_np)

    print("results: ", results)
    
    black_area = np.zeros((400, 10, 3), dtype=np.uint8)

    vis = np.zeros((400,10, 1), dtype=np.unint8)

    for i in range(len(results[0])):

        retrieved_img_index = results[0][i]

        retrieved_img_name = train_database_lines[retrieved_img_index]

        print("retrieved_img_index: ", retrieved_img_index)
        print("retrieved_img_name: ", retrieved_img_name)

        retrieved_img_path = os.path.join(arg.database_img_path, retrieved_img_name)

        retrieved_img = cv2.imread(retrieved_img_path)

        retrieved_img = np.hstack((black_area, retrieved_img))
        vis = np.hstack((vis, retrieved_img))


        #cv2.imshow('{} retrieved result: '.format(i), retrieved_img)
        #cv2.waitKey(0)

    
    cv2.imshow('query img and top 3 retrieved results', vis)
    cv2.waitKey()


if __name__=='__main__':
    main()


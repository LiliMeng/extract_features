'''
Plot boundary of Toronto

Author: Lili Meng
'''

import csv
import pandas as pd 
import random 

import matplotlib
matplotlib.use("TkAgg")
matplotlib.use("qt4agg")
import matplotlib.pyplot as plt 


def sample_data_from_rectangle(sample_ratio=0.1):

	boundary_csv = pd.read_csv('./data/toronto_geo.csv')
	boundary_lng = boundary_csv['longitude']
	boundary_lat = boundary_csv['latitude']

	sample_num = int(sample_ratio*len(data_lng))

	print("sample_num is {}".format(sample_num))

	print(list(zip(data_lng, data_lat)))
	sampled_data_pair = random.sample(list(zip(boundary_lng, boundary_lat)), sample_num)

	print("len(sampled_data_pair): {}".format(len(sampled_data_pair)))

	data_lng_sub = []
	data_lat_sub = []
	for i in range(len(sampled_data_pair)):
		data_lng_sub.append(sampled_data_pair[i][0])
		data_lat_sub.append(sampled_data_pair[i][1])

	return data_lng_sub, data_lat_sub



def sample_data_from_open_house(data_lng, data_lat, sample_ratio=0.1):

	print("num of data_lng: {}, num of data_lat: {}".format(len(data_lng), len(data_lat)))
	sample_num = int(sample_ratio*len(data_lng))

	print("sample_num is {}".format(sample_num))

	print(list(zip(data_lng, data_lat)))
	sampled_data_pair = random.sample(list(zip(data_lng, data_lat)), sample_num)

	print("len(sampled_data_pair): {}".format(len(sampled_data_pair)))

	data_lng_sub = []
	data_lat_sub = []
	for i in range(len(sampled_data_pair)):
		data_lng_sub.append(sampled_data_pair[i][0])
		data_lat_sub.append(sampled_data_pair[i][1])

	return data_lng_sub, data_lat_sub

def remove_data_geo_outlier():

	data_csv = pd.read_csv('./data/open-houses-toronto.csv')
	
	boundary_csv = pd.read_csv('./data/toronto_geo.csv')
	boundary_lng = boundary_csv['longitude']
	boundary_lat = boundary_csv['latitude']
	
	boundary_x_min = min(boundary_lng)
	boundary_x_max = max(boundary_lng)

	boundary_y_min = min(boundary_lat)
	boundary_y_max = max(boundary_lat)

	data_csv = data_csv[(data_csv['longitude'] > boundary_x_min) & (data_csv['longitude']< boundary_x_max)]
	data_csv = data_csv[(data_csv['latitude'] > boundary_y_min) & (data_csv['latitude'] < boundary_y_max)]

	return data_csv

def plot_dataset_and_boundary(data_lng, data_lat, test_ratio=0.4):

	boundary_csv = pd.read_csv('./data/toronto_geo.csv')
	boundary_lng = boundary_csv['longitude']
	boundary_lat = boundary_csv['latitude']

	boundary_x_min = min(boundary_lng)
	boundary_x_max = max(boundary_lng)

	boundary_y_min = min(boundary_lat)
	boundary_y_max = max(boundary_lat)

	train_data_index = int((1-test_ratio)*len(data_lng))

	train_lng = data_lng[0:train_data_index]
	train_lat = data_lat[0:train_data_index]

	test_lng = data_lng[train_data_index:]
	test_lat = data_lat[train_data_index:]

	plt.plot(boundary_lng, boundary_lat, linewidth=2.0)
	
	plt.plot(train_lng, train_lat, 'b.')
	plt.plot(test_lng, test_lat, 'r.')

	
	plt.title('Toronto City Open House Data')
	plt.legend(('Toronto city boundary', 'train data', 'test data'))
	plt.xlim(boundary_x_min, boundary_x_max)
	plt.ylim(boundary_y_min, boundary_y_max)

	plt.xlabel('longitude')
	plt.ylabel('latitude')
	plt.show()
	


if __name__ == '__main__':


	filtered_data = remove_data_geo_outlier()

	data_lng = filtered_data['longitude']
	data_lat = filtered_data['latitude']

	sampling_data = True
	sample_ratio = 0.1

	if sampling_data == True:
		data_lng, data_lat = sample_data(data_lng, data_lat, sample_ratio)
	
	plot_dataset_and_boundary(data_lng, data_lat)

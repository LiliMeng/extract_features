'''
Convex hull and rectangle data sampling 

Author: Lili Meng
'''

import csv
import pandas as pd 
import random 
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
matplotlib.use("qt4agg")
import matplotlib.pyplot as plt 
import scipy.optimize as opt
from scipy.spatial import ConvexHull

random.seed(3)

def find_convex_hull():

	boundary_csv = pd.read_csv('./data/toronto_geo.csv')
	boundary_lng = boundary_csv['longitude']
	boundary_lat = boundary_csv['latitude']

	boundary_points = np.array(list(zip(boundary_lng, boundary_lat)))

	print(boundary_points)

	hull = ConvexHull(boundary_points)

	plt.plot(boundary_points[:,0], boundary_points[:,1], 'o')


	for simplex in hull.simplices:
		plt.plot(boundary_points[simplex, 0], boundary_points[simplex, 1], 'r-')

	plt.title('Toronto City')
	plt.legend(('Toronto city boundary', 'convex hull of the boundary'))
	#plt.xlim(boundary_x_min, boundary_x_max)
	#plt.ylim(boundary_y_min, boundary_y_max)

	plt.xlabel('longitude')
	plt.ylabel('latitude')
	plt.show()

	return boundary_points

def hull_test(P, X, use_hull=True, verbose=False, hull_tolerance=1e-5, return_hull=True):
	'''
	Check a whether given Points P are within a convex hull with convex hull points X
	https://stackoverflow.com/questions/4901959/find-if-a-point-is-inside-a-convex-hull-for-a-set-of-points-without-computing-th
	'''
	if use_hull:
		hull = ConvexHull(X)
		X = X[hull.vertices]

	n_points = len(X)

	def F(x, X, P):
		return np.linalg.norm( np.dot( x.T, X ) - P )

	bnds = [[0, None]]*n_points # coefficients for each point must be > 0
	cons = ( {'type': 'eq', 'fun': lambda x: np.sum(x)-1} ) # Sum of coefficients must equal 1
	x0 = np.ones((n_points,1))/n_points # starting coefficients
	result = opt.minimize(F, x0, args=(X, P), bounds=bnds, constraints=cons)

	if result.fun < hull_tolerance:
		hull_result = True
	else:
		hull_result = False

	if verbose:
		print( '# boundary points:', n_points)
		print( 'x.T * X - P:', F(result.x,X,P) )
		if hull_result:
			print( 'Point P is in the hull space of X')
		else: 
			print( 'Point P is NOT in the hull space of X')

	if return_hull:
		return hull_result, X
	else:
		return hull_result




def sample_data_from_rectangle(num_points=500):

	boundary_csv = pd.read_csv('./data/toronto_geo.csv')
	boundary_lng = boundary_csv['longitude']
	boundary_lat = boundary_csv['latitude']

	boundary_x_min = min(boundary_lng)
	boundary_x_max = max(boundary_lng)

	boundary_y_min = min(boundary_lat)
	boundary_y_max = max(boundary_lat)

	sampled_data_lng = []
	sampled_data_lat = []
	sampled_points = []

	while len(sampled_points) < num_points:
		random_x = random.uniform(boundary_x_min, boundary_x_max)
		random_y = random.uniform(boundary_y_min, boundary_y_max)

		sampled_data_lng.append(random_x)
		sampled_data_lat.append(random_y)

		sampled_points.append([random_x, random_y])

	print(sampled_points)
		
	plt.plot(boundary_lng, boundary_lat, linewidth=2.0)
	plt.plot(sampled_data_lng, sampled_data_lat, 'b.')
	plt.title('Toronto City Open House Data')
	plt.legend(('Toronto city boundary', 'sampled data'))
	plt.xlim(boundary_x_min, boundary_x_max)
	plt.ylim(boundary_y_min, boundary_y_max)
	plt.xlabel('longitude')
	plt.ylabel('latitude')
	plt.show()

	return sampled_points
 
def sample_data_from_convex_hull(num_points=350):

	boundary_csv = pd.read_csv('./data/toronto_geo.csv')
	boundary_lng = boundary_csv['longitude']
	boundary_lat = boundary_csv['latitude']

	boundary_points = np.array(list(zip(boundary_lng, boundary_lat)))

	boundary_x_min = min(boundary_lng)
	boundary_x_max = max(boundary_lng)

	boundary_y_min = min(boundary_lat)
	boundary_y_max = max(boundary_lat)

	sampled_data_lng = []
	sampled_data_lat = []
	sampled_points = []

	while len(sampled_points) < num_points:
		random_x = random.uniform(boundary_x_min, boundary_x_max)
		random_y = random.uniform(boundary_y_min, boundary_y_max)

		hull_result, _ = hull_test([random_x, random_y], boundary_points)
		if hull_result == True:
			sampled_data_lng.append(random_x)
			sampled_data_lat.append(random_y)

			sampled_points.append([random_x, random_y])

	print("num of sampled points is :{}".format(len(sampled_points)))
	hull = ConvexHull(boundary_points)
		
	plt.plot(boundary_lng, boundary_lat, linewidth=2.0)
	plt.plot(sampled_data_lng, sampled_data_lat, 'b.')

	for simplex in hull.simplices:
		plt.plot(boundary_points[simplex, 0], boundary_points[simplex, 1], 'r-')
	

	plt.title('Toronto City Open House Data')
	plt.legend(('Toronto city boundary', 'sampled_data', 'convex hull of boundary'))
	plt.xlim(boundary_x_min, boundary_x_max)
	plt.ylim(boundary_y_min, boundary_y_max)
	plt.xlabel('longitude')
	plt.ylabel('latitude')
	plt.show()

	return sampled_points



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
	

def test_sampling_open_house_data():

	filtered_data = remove_data_geo_outlier()

	data_lng = filtered_data['longitude']
	data_lat = filtered_data['latitude']

	sampling_data = True
	sample_ratio = 0.1

	if sampling_data == True:
		data_lng, data_lat = sample_data(data_lng, data_lat, sample_ratio)
	
	plot_dataset_and_boundary(data_lng, data_lat)


if __name__ == '__main__':

	# test_sampling_open_house_data()
	#hull_points = find_convex_hull()
	num_points = 350
	#sampled_points = sample_data_from_rectangle(num_points)
	sampled_points = sample_data_from_convex_hull(num_points)

	print("sampled_points: {}".format(sampled_points))

	
    

'''
Query and save Google places type data
Author: Lili Meng
'''

from data_sampling_vis import sample_data_from_convex_hull
from google_placesAPI import nearby_search_url
from six.moves import urllib

import requests
import json
import os 


def save_url_as_json(url, save_file_name):

	urllib.request.urlretrieve(url, save_file_name)
	

if __name__ == '__main__':


	num_points = 350
	
	sampled_points = sample_data_from_convex_hull(num_points)

	#print("sampled_points: {}".format(sampled_points))
	search_radius = 2000

	place_type = 'restaurant'

	saved_json_foldername = 'saved_json'

	if not os.path.exists(saved_json_foldername):
		os.makedirs(saved_json_foldername)

	for i in range(3):
		url = nearby_search_url(sampled_points[i][0], sampled_points[i][1], search_radius, place_type)
		save_file_name = os.path.join(saved_json_foldername, 'data_{}.json'.format('%05d'%i))
		save_url_as_json(url, save_file_name)


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import configparser
import codecs
from distutils.util import strtobool
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples

import sys
sys.path.append("../MyPythonModule")
import mymodule
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx

"""保存名の決定（root_dir）"""
def suffix_generator_root(search_word,max_page,add_childs,append):
	suffix = "_" + search_word
	suffix += "_" + unicode(max_page)
	if add_childs:
		suffix += "_add_childs"
	if append:
		suffix += "_append"
	return suffix

"""保存名の決定"""
def suffix_generator(target=None,is_largest=False):
	suffix = ""
	if target != None:
		suffix += "_" + target
	if is_largest == True:
		suffix += "_largest"
	return suffix

COLORLIST_R = [r"#EB6100",r"#F39800",r"#FCC800",r"#FFF100",r"#CFDB00",r"#8FC31F",r"#22AC38",r"#009944",r"#009B6B",r"#009E96",r"#00A0C1",r"#00A0E9",r"#0086D1",r"#0068B7",r"#00479D",r"#1D2088",r"#601986",r"#920783",r"#BE0081",r"#E4007F",r"#E5006A",r"#E5004F",r"#E60033"]
COLORLIST = [c for c in COLORLIST_R[::2]]

if __name__ == "__main__":
	"""設定ファイルの読み込み"""
	inifile = configparser.ConfigParser(allow_no_value = True,interpolation = configparser.ExtendedInterpolation())
	inifile.readfp(codecs.open("./series_act.ini",'r','utf8'))

	"""検索パラメータの設定"""
	search_word = inifile.get('options','search_word')
	max_page = int(inifile.get('options','max_page'))
	add_childs = strtobool(inifile.get('options','add_childs'))
	append = strtobool(inifile.get('options','append'))
	save_dir = inifile.get('other_settings','save_dir')
	root_dir = save_dir + suffix_generator_root(search_word,max_page,add_childs,append)

	is_largest = strtobool(inifile.get('options','is_largest'))
	target = inifile.get('options','target')

	size = int(inifile.get('d2v','size'))
	exp_name_new = "D" + str(size) + suffix_generator(target,is_largest)
	exp_dir_new = os.path.join(root_dir,exp_name_new)
	nx_dir_new = os.path.join(exp_dir_new,"nx_datas")
	check_k = os.path.join(nx_dir_new,"check_k")

	if not os.path.exists(check_k):
		os.mkdir(check_k)

	"""ファイルの読み込み"""
	with open(os.path.join(exp_dir_new,"doc2vec.pkl"),'rb') as fi:
		doc2vec_vectors = pickle.load(fi)

	print("ノード数：" + str(len(doc2vec_vectors.keys())))

	data_array = doc2vec_vectors.values()

	"""エルボー法"""
	distortions_10 = []
	for i  in range(1,11):
		km = KMeans(n_clusters=i,random_state=0)
		km.fit(data_array)
		distortions_10.append(km.inertia_)

	plt.plot(range(1,11),distortions_10,marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Distortion')
	plt.savefig(os.path.join(check_k,"elbow10.png"))
	# plt.show()

	distortions_100 = []
	for i  in range(1,101):
		km = KMeans(n_clusters=i,random_state=0)
		km.fit(data_array)
		distortions_100.append(km.inertia_)

	plt.plot(range(1,101),distortions_100,marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Distortion')
	plt.savefig(os.path.join(check_k,"elbow100.png"))
	# plt.show()

	"""シルエット分析"""
	# n_clusters = 10
	# km = KMeans(n_clusters=n_clusters,random_state=0)
	# y_km = km.fit_predict(data_array)
	#
	# cluster_labels = np.unique(y_km)
	# n_clusters=cluster_labels.shape[0]
	# silhouette_vals = silhouette_samples(data_array,y_km,metric='euclidean')
	# y_ax_lower, y_ax_upper = 0,0
	# yticks = []
	#
	# for i,c in enumerate(cluster_labels):
	# 	c_silhouette_vals = silhouette_vals[y_km==c]
	# 	c_silhouette_vals.sort()
	# 	y_ax_upper += len(c_silhouette_vals)
	# 	color = cm.jet(float(i)/n_clusters)
	# 	plt.barh(range(y_ax_lower,y_ax_upper),c_silhouette_vals,height=1.0,edgecolor='none',color=color)
	# 	yticks.append((y_ax_lower+y_ax_upper)/2)
	# 	y_ax_lower += len(c_silhouette_vals)
	#
	# silhouette_avg = np.mean(silhouette_vals)
	# plt.axvline(silhouette_avg,color="red",linestyle="--")
	# plt.yticks(yticks,cluster_labels + 1)
	# plt.ylabel('Cluster')
	# plt.xlabel('silhouette coefficient')
	# plt.show()

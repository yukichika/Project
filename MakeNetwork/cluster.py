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

	if not os.path.exists(nx_dir_new):
		print("D2V modify not finished.")
	else:
		"""ファイルの読み込み"""
		with open(os.path.join(exp_dir_new,"doc2vec.pkl"),'rb') as fi:
			doc2vec_vectors = pickle.load(fi)
		with open(os.path.join(nx_dir_new,"G_with_params_cos_sim.gpkl"),'rb') as fi:
			G = pickle.load(fi)
		with open(os.path.join(nx_dir_new,"G_with_params_euclid.gpkl"),'rb') as fi:
			G_euc = pickle.load(fi)

		if G.node.keys() == doc2vec_vectors.keys():
			print("ノード数：" + str(len(G.node.keys())))

			data_array = doc2vec_vectors.values()

			pca = decomposition.PCA(3)
			pca.fit(data_array)
			data_array_pca = pca.transform(data_array)

			n_clusters = 10
			"""ベクトルの次元数を保持したままクラスタリング"""
			print("-----ベクトルの次元数を保持したままクラスタリング-----")
			kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(data_array)
			pred = kmeans.labels_
			center_100 = kmeans.cluster_centers_
			# print(kmeans.cluster_centers_)
			center_100_dict = {}
			for i,c_100 in enumerate(center_100):
				center_100_dict[i] = c_100
			# print(center_100_dict)

			u, c = np.unique(pred,return_counts=True)
			result_sum = dict(zip(u, c))
			print(result_sum)

			with open(os.path.join(nx_dir_new,"kmeans_n" + str(n_clusters) + "_d" + str(len(doc2vec_vectors[0])) + ".txt"),'w') as fo:
				fo.write("クラスタ：総数" + "\n")
				for k,v in result_sum.items():
					fo.write(str(k) + ":" + str(v) + "\n")

			"""ベクトルの次元数を主成分分析で圧縮してクラスタリング"""
			print("-----ベクトルの次元数を主成分分析で圧縮してクラスタリング-----")
			kmeans_pca = KMeans(n_clusters=n_clusters,random_state=0).fit(data_array_pca)
			pred_pca = kmeans_pca.labels_
			center_3 = kmeans_pca.cluster_centers_
			# print(kmeans_pca.cluster_centers_)
			center_3_dict = {}
			for i,c_3 in enumerate(center_3):
				center_3_dict[i] = c_3
			# print(center_3_dict)

			u, c = np.unique(pred_pca,return_counts=True)
			result_sum = dict(zip(u, c))
			print(result_sum)

			with open(os.path.join(nx_dir_new,"kmeans_n" + str(n_clusters) + "_d" + str(len(data_array_pca[0])) + ".txt"),'w') as fo:
				fo.write("クラスタ：総数" + "\n")
				for k,v in result_sum.items():
					fo.write(str(k) + ":" + str(v) + "\n")

			"""グラフに反映"""
			nodes = G.node
			nodes_euc = G_euc.node
			for node_no,p,p_pca in zip(G.node.keys(),pred,pred_pca):
				kmeans_100 = p
				kmeans_3 = p_pca

				nodes[node_no]["kmeans_100"] = kmeans_100
				nodes[node_no]["kmeans_3"] = kmeans_3
				nodes[node_no]["color_k3"] = COLORLIST[kmeans_3]
				nodes[node_no]["color_k100"] = COLORLIST[kmeans_100]

				nodes_euc[node_no]["kmeans_100"] = kmeans_100
				nodes_euc[node_no]["kmeans_3"] = kmeans_3
				nodes_euc[node_no]["color_k3"] = COLORLIST[kmeans_3]
				nodes_euc[node_no]["color_k100"] = COLORLIST[kmeans_100]

			"""データの書き出し"""
			with open(os.path.join(nx_dir_new,"G_with_params_cos_sim.gpkl"),'w') as fo:
				pickle.dump(G,fo)
			with open(os.path.join(nx_dir_new,"G_with_params_euclid.gpkl"),'w') as fo:
				pickle.dump(G_euc,fo)

			with open(os.path.join(nx_dir_new,"kmeans_n" + str(n_clusters) + "_d" + str(len(doc2vec_vectors[0])) + ".pkl"),'w') as fo:
				pickle.dump(center_100_dict,fo)
			with open(os.path.join(nx_dir_new,"kmeans_n" + str(n_clusters) + "_d" + str(len(data_array_pca[0])) + ".pkl"),'w') as fo:
				pickle.dump(center_3_dict,fo)

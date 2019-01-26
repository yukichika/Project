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
import itertools

import sys
sys.path.append("../MyPythonModule")
import mymodule
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx

"""ユークリッド距離"""
def euclid(p,q):
	weight = np.sqrt(np.power(p-q,2).sum())
	return weight

"""ベクトルのノルム"""
def norm(p):
	vec_norm = np.linalg.norm(p)
	return vec_norm

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
	check_sort = os.path.join(nx_dir_new,"check_sort")
	if not os.path.exists(check_sort):
		os.mkdir(check_sort)

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

		with open(os.path.join(nx_dir_new,"kmeans_n10_d100.pkl"),'rb') as fi:
			dict_100 = pickle.load(fi)
		with open(os.path.join(nx_dir_new,"kmeans_n10_d3.pkl"),'rb') as fi:
			dict_3 = pickle.load(fi)

		"""ターゲットを定めてソート"""
		# target = 0
		# distance_list = []
		# for i in dict_100.keys():
		# 	if not i == target:
		# 		distance = euclid(dict_100[target],dict_100[i])
		# 		distance_list.append([i,distance])
		#
		# distance_list = sorted(distance_list,key=lambda x: x[1],reverse=False)
		# with open(os.path.join(check_sort,"target_" + str(target) + ".txt"),'w') as fo:
		# 	fo.write("target cluster number:" + str(target) + "\n")
		# 	fo.write("(old,new,distance)" + "\n")
		#
		# 	for old,new in zip(dict_100.keys(),distance_list):
		# 		fo.write("(" + str(old) + "," + str(new[0]) + "," + str(new[1]) + ")" + "\n")

		"""全組み合わせ"""
		# distance_list = []
		# for comb in list(itertools.combinations(dict_100.keys(),2)):
		# 	distance = euclid(dict_100[comb[0]],dict_100[comb[1]])
		# 	distance_list.append([(comb[0],comb[1]),distance])
		# distance_list = sorted(distance_list,key=lambda x: x[1],reverse=False)
		# print(distance_list)

		"""原点からの距離（ノルム）に応じてソート"""
		# distance_list = []
		# for k,v in dict_100.items():
		# 	norm_vec = norm(v)
		# 	distance_list.append([k,norm_vec])
		#
		# distance_list = sorted(distance_list,key=lambda x: x[1],reverse=False)
		# print(distance_list)
		#
		# with open(os.path.join(check_sort,"origin.txt"),'w') as fo:
		# 	fo.write("(old,new,distance)" + "\n")
		#
		# 	for old,new in zip(dict_100.keys(),distance_list):
		# 		fo.write("(" + str(old) + "," + str(new[0]) + "," + str(new[1]) + ")" + "\n")

		"""重心点を主成分分析し，カラーバーに割り当て"""
		# data = [dict_100[i] for i in dict_100.keys()]
		#
		# vecs = [doc2vec_vectors[x] for x in doc2vec_vectors.keys()]
		# pca = decomposition.PCA(1)
		# pca.fit(vecs)
		# vecs_pca = pca.transform(data)
		# reg_vecs_pca = (vecs_pca-vecs_pca.min())/(vecs_pca.max()-vecs_pca.min())#0~1に正規化
		# sorted_reg_vecs_pca = sorted(reg_vecs_pca,key=lambda x: x[0],reverse=False)
		#
		# result = []
		# for new_rank,new in enumerate(sorted_reg_vecs_pca):
		# 	for old_rank,old in enumerate(reg_vecs_pca):
		# 		if new == old:
		# 			result.append([(new_rank,old_rank),new])
		#
		# print(result)
		# with open(os.path.join(check_sort,"pca1.txt"),'w') as fo:
		# 	fo.write("(new,old)" + "\n")
		#
		# 	for r in result:
		# 		fo.write("(" + str(r[0][0]) + "," + str(r[0][1]) + ")" + "\n")

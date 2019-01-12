#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import configparser
import codecs
from distutils.util import strtobool
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx

"""コサイン類似度"""
def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

"""指数で正規化したユークリッド距離"""
def compare4_2(p,q):
	weight = np.exp(-((p-q)**2).sum())
	return weight

"""ユークリッド距離"""
def euclid(p,q):
	weight = np.sqrt(np.power(p-q,2).sum())
	return weight

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
	K = int(inifile.get('lda','K'))
	exp_name = "K" + unicode(K) + suffix_generator(target,is_largest)
	exp_dir = os.path.join(root_dir,exp_name)
	nx_dir = os.path.join(exp_dir,"nx_datas")

	size = int(inifile.get('d2v','size'))
	exp_name_new = "D" + str(size) + suffix_generator(target,is_largest)
	exp_dir_new = os.path.join(root_dir,exp_name_new)
	nx_dir_new = os.path.join(exp_dir_new,"nx_datas")
	check_dir = os.path.join(nx_dir_new,"check_distance_distribution")

	if not os.path.exists(check_dir):
		os.mkdir(check_dir)

	cos = []#コサイン類似度
	cos_norm = []#コサイン類似度正規化(0~1)
	euclid_ = []#ユークリッド距離
	euclid_norm = []#ユークリッド距離正規化(0~1)
	euclid_norm_reverse = []#反転させたユークリッド距離正規化(0~1)
	euclid_index = []#指数で正規化したユークリッド距離

	comp_func_name = inifile.get('nx','comp_func_name')
	G_path = "G_with_params_" + comp_func_name + ".gpkl"

	"""ファイルの読み込み"""
	# with open(os.path.join(exp_dir,"instance.pkl"),'r') as fi:
	# 	lda = pickle.load(fi)
	with open(os.path.join(nx_dir,G_path),'r') as fi:
		G = pickle.load(fi)
	with open(os.path.join(exp_dir_new,"doc2vec.pkl"),'rb') as fi:
		doc2vec_vectors = pickle.load(fi)

	"""0~1に正規化するために最大値・最小値の取得（自分自身は除く）"""
	nodes = G.node
	for i,i_node in enumerate(tqdm(nodes)):
		p_dst = doc2vec_vectors[i_node]
		for j,j_node in enumerate(nodes):
			if not j == i:
				q_dst = doc2vec_vectors[j_node]
				weight_cos = cos_sim(p_dst,q_dst)
				weight_euclid = euclid(p_dst,q_dst)
				weight_euclid_index = compare4_2(p_dst,q_dst)
				cos.append(weight_cos)
				euclid_.append(weight_euclid)
				euclid_index.append(weight_euclid_index)
	max_cos = np.array(cos).max()
	min_cos = np.array(cos).min()
	max_euclid = np.array(euclid_).max()
	min_euclid = np.array(euclid_).min()

	"""全ノード間距離算出(自分自身は除く)"""
	for i,i_node in enumerate(tqdm(nodes)):
		p_dst = doc2vec_vectors[i_node]
		for j,j_node in enumerate(nodes):
			if not j == i:
				q_dst = doc2vec_vectors[j_node]
				weight_cos = cos_sim(p_dst,q_dst)
				weight_cos = (weight_cos - min_cos)/(max_cos - min_cos)#0~1に正規化
				cos_norm.append(weight_cos)
				weight_euclid = euclid(p_dst,q_dst)
				weight_euclid = (weight_euclid - min_euclid)/(max_euclid - min_euclid)#0~1に正規化
				euclid_norm.append(weight_euclid)
				weight_euclid_reverse = 1 - weight_euclid
				euclid_norm_reverse.append(weight_euclid_reverse)


	"""weight（全ノード間の距離）のヒストグラム作成（コサイン類似度）"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(cos,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.8, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
	plt.text(0.8, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
	plt.text(0.8, 0.80, "total="+str(len(cos)), transform=ax.transAxes)
	fig_w.show()
	fig_w.savefig(os.path.join(check_dir,"cos.png"))

	"""weight（全ノード間の距離）のヒストグラム作成（正規化したコサイン類似度）"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(cos_norm,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.8, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
	plt.text(0.8, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
	plt.text(0.8, 0.80, "total="+str(len(cos_norm)), transform=ax.transAxes)
	fig_w.show()
	fig_w.savefig(os.path.join(check_dir,"cos_norm.png"))

	"""weight（全ノード間の距離）のヒストグラム作成（ユークリッド距離）"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(euclid_,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.8, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
	plt.text(0.8, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
	plt.text(0.8, 0.80, "total="+str(len(euclid_)), transform=ax.transAxes)
	plt.xlim([weights_array.min(),weights_array.max()])
	fig_w.show()
	fig_w.savefig(os.path.join(check_dir,"euclid.png"))

	"""weight（全ノード間の距離）のヒストグラム作成（正規化したユークリッド距離）"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(euclid_norm,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.8, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
	plt.text(0.8, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
	plt.text(0.8, 0.80, "total="+str(len(euclid_norm)), transform=ax.transAxes)
	plt.xlim([weights_array.min(),weights_array.max()])
	fig_w.show()
	fig_w.savefig(os.path.join(check_dir,"euclid_norm.png"))

	"""weight（全ノード間の距離）のヒストグラム作成（反転させた正規化したユークリッド距離）"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(euclid_norm_reverse,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.8, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
	plt.text(0.8, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
	plt.text(0.8, 0.80, "total="+str(len(euclid_norm_reverse)), transform=ax.transAxes)
	plt.xlim([weights_array.min(),weights_array.max()])
	fig_w.show()
	fig_w.savefig(os.path.join(check_dir,"euclid_norm_reverse.png"))

	"""weight（全ノード間の距離）のヒストグラム作成（指数で正規化したユークリッド距離）"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(euclid_index,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.5, 0.9, "max="+str(weights_array.max()), transform=ax.transAxes)
	plt.text(0.5, 0.85, "min="+str(weights_array.min()), transform=ax.transAxes)
	plt.text(0.5, 0.80, "total="+str(len(euclid_index)), transform=ax.transAxes)
	plt.xlim([weights_array.min(),weights_array.max()])
	fig_w.show()
	fig_w.savefig(os.path.join(check_dir,"euclid_index.png"))

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
sys.path.append("../MyPythonModule")
import mymodule
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx

"""コサイン類似度"""
def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

"""指数で正規化したユークリッド距離"""
def compare4_2(p,q):
	weight = np.exp(-((p-q)**2).sum())
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

	if os.path.exists(nx_dir_new):
		print("D2V modify finished.")
	else:
		os.mkdir(nx_dir_new)

		weights_list = []
		weights_list_new = []
		euclids = []

		comp_func_name = inifile.get('nx','comp_func_name')
		G_path = "G_with_params_" + comp_func_name + ".gpkl"

		"""ファイルの読み込み"""
		with open(os.path.join(exp_dir,"instance.pkl"),'r') as fi:
			lda = pickle.load(fi)
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
					weight = cos_sim(p_dst,q_dst)
					weights_list.append(weight)
		max = np.array(weights_list).max()
		min = np.array(weights_list).min()

		"""エッジ間の距離算出（ユークリッド距離=>コサイン類似度に変更）"""
		edges = G.edge
		for node_no,link_node_nos in edges.items():
			p_dst = doc2vec_vectors[node_no]
			"""類似度による重みの算出"""
			for link_node_no in link_node_nos.keys():
				q_dst = doc2vec_vectors[link_node_no]
				weight = cos_sim(p_dst,q_dst)
				weight = (weight - min)/(max - min)#0~1に正規化
				if weight == 0:
					weight = 0.001
				edges[node_no][link_node_no]["weight"] = weight

		DEFAULT_WEIGHT = 0.5
		"""全ノード間距離算出．上といろいろ重複するが面倒なのでもう一度ループ（コサイン類似度）"""
		nodes = G.node
		nodes_lim = len(nodes)
		all_node_weights = np.ones((nodes_lim,nodes_lim))*DEFAULT_WEIGHT
		for i,i_node in enumerate(tqdm(nodes)):
			p_dst = doc2vec_vectors[i_node]
			for j,j_node in enumerate(nodes):
				q_dst = doc2vec_vectors[j_node]
				weight = cos_sim(p_dst,q_dst)
				weight = (weight - min)/(max - min)#0~1に正規化
				weight_euclid = compare4_2(p_dst,q_dst)
				if weight == 0:
					weight = 0.001
				all_node_weights[i,j] = weight#自分自身との類似度は1を超えるが，使用しないため問題なし
				if not j == i:
					weights_list_new.append(weight)#ヒストグラム作成用
					euclids.append(weight_euclid)#ヒストグラム作成用

		"""データの書き出し"""
		with open(os.path.join(nx_dir_new,"G_with_params_cos_sim.gpkl"),'w') as fo:
			pickle.dump(G,fo)
		with open(os.path.join(nx_dir_new,"all_node_weights_cos_sim.gpkl"),'w') as fo:
			pickle.dump(all_node_weights,fo)

		"""weight（全ノード間の距離）のヒストグラム作成（コサイン類似度）"""
		fig_w = plt.figure()
		ax = fig_w.add_subplot(1,1,1)
		weights_array = np.array(weights_list_new,dtype=np.float)
		ax.hist(weights_array,bins=100)
		plt.text(0.5, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
		plt.text(0.5, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
		plt.text(0.5, 0.80, "num="+str(len(weights_list_new)), transform=ax.transAxes)
		fig_w.show()
		fig_w.savefig(os.path.join(nx_dir_new,"cos_sim_hist.png"))

		"""weight（全ノード間の距離）のヒストグラム作成（指数で正規化したユークリッド距離）"""
		fig_w = plt.figure()
		ax = fig_w.add_subplot(1,1,1)
		weights_array = np.array(euclids,dtype=np.float)
		ax.hist(weights_array,bins=100)
		# plt.text(0.5, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
		# plt.text(0.5, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
		plt.text(0.5, 0.9, "max="+str(weights_array.max()), transform=ax.transAxes)
		plt.text(0.5, 0.85, "min="+str(weights_array.min()), transform=ax.transAxes)
		plt.text(0.5, 0.80, "num="+str(len(euclids)), transform=ax.transAxes)
		plt.xlim([weights_array.min(),weights_array.max()])
		fig_w.show()
		fig_w.savefig(os.path.join(nx_dir_new,"comp4_2_hist.png"))

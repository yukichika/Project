#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import configparser
import codecs
from distutils.util import strtobool
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("../MyPythonModule")
import mymodule
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx

# import cvt_to_nxtype2
# import LDA_for_SS
# import LDA_modify_for_graph
# import calc_HITS
# import make_network_by_nx
# import clollection_analizer

"""コサイン類似度"""
def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

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
	inifile.readfp(codecs.open("./D2V.ini",'r','utf8'))

	"""検索パラメータの設定"""
	search_word = inifile.get('options','search_word')
	max_page = int(inifile.get('options','max_page'))
	add_childs = strtobool(inifile.get('options','add_childs'))
	append = strtobool(inifile.get('options','append'))
	save_dir = inifile.get('other_settings','save_dir')
	root_dir = save_dir + suffix_generator_root(search_word,max_page,add_childs,append)
	Myexttext_pre = os.path.join(root_dir,"Myexttext_pre")

	is_largest = strtobool(inifile.get('options','is_largest'))
	target = inifile.get('options','target')
	K = int(inifile.get('lda','K'))
	exp_name = "K" + unicode(K) + suffix_generator(target,is_largest)

	comp_func_name = inifile.get('nx','comp_func_name')

	exp_dir = os.path.join(root_dir,exp_name)
	nx_dir = os.path.join(exp_dir,"nx_datas")

	weights_list = []
	G_path = "G_with_params_" + comp_func_name + ".gpkl"

	"""ファイルの読み込み"""
	with open(os.path.join(exp_dir,"instance.pkl"),'r') as fi:
		lda = pickle.load(fi)
	with open(os.path.join(exp_dir,"doc2vec.pkl"),'rb') as fi:
		d2v = pickle.load(fi)
	with open(os.path.join(nx_dir,G_path),'r') as fi:
		G = pickle.load(fi)

	file_id_dict_inv = {v:k for k, v in lda.file_id_dict.items()}

	"""エッジ間の距離算出"""
	edges = G.edge
	for node_no,link_node_nos in edges.items():
		p_dst = d2v[node_no]
		"""類似度による重みの算出"""
		for link_node_no in link_node_nos.keys():
			q_dst = d2v[link_node_no]
			weight = cos_sim(p_dst,q_dst)
			edges[node_no][link_node_no]["weight"] = weight

	DEFAULT_WEIGHT = 0.5
	"""全ノード間距離算出．上といろいろ重複するが面倒なのでもう一度ループ"""
	nodes = G.node
	nodes_lim = len(nodes)#removeしている場合があるため
	print(nodes_lim)
	all_node_weights = np.ones((nodes_lim,nodes_lim))*DEFAULT_WEIGHT#除算の都合上，自分自身との類似度は1に
	for i,i_node in enumerate(tqdm(nodes)):
		p_dst = d2v[i_node]
		for j,j_node in enumerate(nodes):
			q_dst = d2v[j_node]
			weight = cos_sim(p_dst,q_dst)
			if weight == 0:
				weight = 0.001
			all_node_weights[i,j] = weight
			#all_node_weights[i,j] = all_node_weights[j,i]=weight
			weights_list.append(weight)#ヒストグラム作成用

	"""データの書き出し"""
	with open(os.path.join(nx_dir,"G_with_params_cos_sim.gpkl"),'w') as fo:
		pickle.dump(G,fo)
	with open(os.path.join(nx_dir,"all_node_weights_cos_sim.gpkl"),'w') as fo:
		pickle.dump(all_node_weights,fo)

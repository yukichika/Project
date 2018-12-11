#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import configparser
import codecs
from distutils.util import strtobool
import MeCab

import sys
sys.path.append("../../MyPythonModule")
import mymodule
sys.path.append("../../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx

# import cvt_to_nxtype2
# import LDA_for_SS
# import LDA_modify_for_graph
# import calc_HITS
# import make_network_by_nx
# import clollection_analizer

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

def words_list(text):
	mecab = MeCab.Tagger("-Ochasen")
	lines = mecab.parse(text).splitlines()
	allwords = []
	for line in lines:
		chunks = line.split('\t')
		if not chunks[0] == "\ufeff":#UTF-8の識別子\ufeff
			allwords.append(chunks[0])
	return allwords

def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

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
	with open(os.path.join(exp_dir,"instance.pkl")) as fi:
	   lda = pickle.load(fi)
	with open(os.path.join(nx_dir,G_path)) as fi:
		G = pickle.load(fi)

	file_id_dict_inv = {v:k for k, v in lda.file_id_dict.items()}

	DEFAULT_WEIGHT = 0.5
	"""エッジ間の距離算出"""
	edges = G.edge
	for node_no,link_node_nos in edges.items():
		vec_no = file_id_dict_inv.get(node_no)#ファイルが連番でない対象に対応
		p_dst = vectors[vec_no]
		"""類似度による重みの算出"""
		for link_node_no in link_node_nos.keys():
			link_vec_no = file_id_dict_inv.get(link_node_no)#ファイルが連番でない対象に対応
			q_dst = theta[link_vec_no]
			weight = compare(p_dst,q_dst)
			edges[node_no][link_node_no]["weight"] = weight

		# """ベクトル化"""
		# INPUT_MODEL = u"/home/yukichika/ドキュメント/Doc2vec_model/Wikipedia809710_dm_100_w5_m5_20.model"
		# model = models.Doc2Vec.load(INPUT_MODEL)
	    #
		# vectors = []
		# targets = os.listdir(Myexttext_pre)
		# Mymodule.sort_nicely(targets)
		# for file in targets:
		# 	node_no = file.split(".")[0]
		# 	with open(os.path.join(Myexttext_pre,file),'r') as fi:
		# 		text = fi.read()
		# 		words = Wakati.words_list(text)
		# 		vector = model.infer_vector(words)
		# 		print(len(vector))

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
		with open(os.path.join(nx_dir_new,"G_with_params_cos_sim.gpkl"),'rb') as fi:
			G = pickle.load(fi)
		with open(os.path.join(nx_dir_new,"G_with_params_euclid.gpkl"),'rb') as fi:
			G_euc = pickle.load(fi)

		with open(os.path.join(exp_dir_new,"kmeans_n10_d100.pkl"),'rb') as fi:
			dict_100 = pickle.load(fi)
		with open(os.path.join(exp_dir_new,"kmeans_n10_d3.pkl"),'rb') as fi:
			dict_3 = pickle.load(fi)

		print(dict_100)
		print(G.node[0])

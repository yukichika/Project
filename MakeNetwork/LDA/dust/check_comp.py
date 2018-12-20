#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import configparser
import codecs
from distutils.util import strtobool
from tqdm import tqdm
from gensim import models
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../MyPythonModule")
from LDA_kai import LDA

"""保存名の決定"""
def suffix_generator_root(search_word,max_page,add_childs,append):
	suffix = "_" + search_word
	suffix += "_" + str(max_page)
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

"""指数で正規化したユークリッド距離"""
def compare4_2(p,q):
	weight = np.exp(-((p-q)**2).sum())
	return weight

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
	exp_name = "K" + str(K) + suffix_generator(target,is_largest)
	exp_dir = os.path.join(root_dir,exp_name)

	with open(os.path.join(exp_dir,"instance.pkl"),'r') as fi:
		lda = pickle.load(fi)

	file_id_dict_inv = {v:k for k, v in lda.file_id_dict.items()}
	node_list = file_id_dict_inv.keys()

	euclids = []
	for node in node_list:
		p = lda.theta()[file_id_dict_inv[node]]
		for node in node_list:
			q = lda.theta()[file_id_dict_inv[node]]
			euclids.append(compare4_2(p,q))

	"""指数で正規化したユークリッド距離のヒストグラム作成"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(euclids,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.5, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
	plt.text(0.5, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
	fig_w.show()
	fig_w.savefig(os.path.join(exp_dir,"check_euclid_hist.png"))

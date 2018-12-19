#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""doc2vecのベクトルの類似度のヒストグラム作成用"""

import os
import cPickle as pickle
import configparser
import codecs
from distutils.util import strtobool
from tqdm import tqdm
from gensim import models
import random
import numpy as np
import matplotlib.pyplot as plt

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

"""コサイン類似度"""
def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

"""指数で正規化したユークリッド距離"""
def compare4_2(p,q):
	weight = np.exp(-((p-q)**2).sum())
	return weight

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

	is_largest = strtobool(inifile.get('options','is_largest'))
	target = inifile.get('options','target')
	size = int(inifile.get('lda','size'))
	exp_name_new = "D" + str(size) + suffix_generator(target,is_largest)
	exp_dir_new = os.path.join(root_dir,exp_name_new)

	with open(os.path.join(exp_dir_new,"doc2vec.pkl"),'r') as fi:
		d2v = pickle.load(fi)

	node_list = d2v.keys()
	random_node = random.choice(node_list)
	p = d2v[random_node]

	cos_sims = []
	euclids = []
	for node in node_list:
		if not node == random_node:
			q = d2v[node]
			cos_sims.append(cos_sim(p,q))
			euclids.append(compare4_2(p,q))

	"""コサイン類似度のヒストグラム作成"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(cos_sims,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.5, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
	plt.text(0.5, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
	plt.text(0.5, 0.80, "node_no="+"{0}".format(random_node), transform=ax.transAxes)
	fig_w.show()
	fig_w.savefig(os.path.join(exp_dir_new,"check_cos_sim_hist.png"))

	"""指数で正規化したユークリッド距離のヒストグラム作成"""
	fig_w = plt.figure()
	ax = fig_w.add_subplot(1,1,1)
	weights_array = np.array(euclids,dtype=np.float)
	ax.hist(weights_array,bins=100)
	plt.text(0.5, 0.9, "max="+"{0:.3f}".format(weights_array.max()), transform=ax.transAxes)
	plt.text(0.5, 0.85, "min="+"{0:.3g}".format(weights_array.min()), transform=ax.transAxes)
	plt.text(0.5, 0.80, "node_no="+"{0}".format(random_node), transform=ax.transAxes)
	fig_w.show()
	fig_w.savefig(os.path.join(exp_dir_new,"check_euclid_hist.png"))

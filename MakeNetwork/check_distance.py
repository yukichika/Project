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
import json

import sys
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx
sys.path.append("../MyPythonModule")
import mymodule

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
	check_dir = os.path.join(nx_dir_new,"check_distance")

	if not os.path.exists(check_dir):
		os.mkdir(check_dir)

	comp_func_name = inifile.get('nx','comp_func_name')
	G_path = "G_with_params_" + comp_func_name + ".gpkl"

	"""ファイルの読み込み"""
	# with open(os.path.join(exp_dir,"instance.pkl"),'r') as fi:
	# 	lda = pickle.load(fi)
	with open(os.path.join(nx_dir,G_path),'r') as fi:
		G = pickle.load(fi)
	with open(os.path.join(exp_dir_new,"doc2vec.pkl"),'rb') as fi:
		doc2vec_vectors = pickle.load(fi)

	cos = []#コサイン類似度
	euclid_ = []#ユークリッド距離
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
				# euclid_index.append(weight_euclid_index)
	max_cos = np.array(cos).max()
	min_cos = np.array(cos).min()
	max_euclid = np.array(euclid_).max()
	min_euclid = np.array(euclid_).min()

	cos_norm = []#コサイン類似度正規化(0~1)
	euclid_norm = []#ユークリッド距離正規化(0~1)
	euclid_index = []#指数で正規化したユークリッド距離

	target_no = 1
	topn = 50

	p_dst = doc2vec_vectors[target_no]
	for i,i_node in enumerate(tqdm(nodes)):
		q_dst = doc2vec_vectors[i_node]
		weight_cos = cos_sim(p_dst,q_dst)
		weight_cos = (weight_cos - min_cos)/(max_cos - min_cos)#0~1に正規化
		weight_euclid = euclid(p_dst,q_dst)
		weight_euclid = (weight_euclid - min_euclid)/(max_euclid - min_euclid)#0~1に正規化
		weight_index = compare4_2(p_dst,q_dst)

		cos_norm.append([i_node,weight_cos])
		euclid_norm.append([i_node,weight_euclid])
		euclid_index.append([i_node,weight_index])

	cos_norm = sorted(cos_norm,key=lambda x: x[1],reverse=True)
	euclid_norm = sorted(euclid_norm,key=lambda x: x[1],reverse=False)
	euclid_index = sorted(euclid_index,key=lambda x: x[1],reverse=True)

	src_pages_dir = os.path.join(root_dir,"Pages")
	with open(os.path.join(src_pages_dir,str(target_no) + ".json")) as fj:
		tgt_json_dict = json.load(fj)
	tgt_title = tgt_json_dict.get("title","")

	with open(os.path.join(check_dir,str(target_no) + "_cos.txt"),'w') as f_cos:
		f_cos.write(tgt_title.encode("utf-8") + "\n\n")
		for i,c in enumerate(cos_norm):
			with open(os.path.join(src_pages_dir,str(c[0]) + ".json")) as fj:
				src_json_dict = json.load(fj)
			src_title = src_json_dict.get("title","")

			f_cos.write(str(c[0]) + " " + src_title.encode("utf-8") + ":" + str(c[1]) + "\n")
			if i+1 == topn:
				break

	with open(os.path.join(check_dir,str(target_no) + "_euclid.txt"),'w') as f_euclid:
		f_euclid.write(tgt_title.encode("utf-8") + "\n\n")
		for j,e in enumerate(euclid_norm):
			with open(os.path.join(src_pages_dir,str(e[0]) + ".json")) as fj:
				src_json_dict = json.load(fj)
			src_title = src_json_dict.get("title","")

			f_euclid.write(str(e[0]) + " " + src_title.encode("utf-8") + ":" + str(e[1]) + "\n")
			if j+1 == topn:
				break

	with open(os.path.join(check_dir,str(target_no) + "_euclid_indec.txt"),'w') as f_euclid_index:
		f_euclid_index.write(tgt_title.encode("utf-8") + "\n\n")
		for k,e_index in enumerate(euclid_index):
			with open(os.path.join(src_pages_dir,str(e_index[0]) + ".json")) as fj:
				src_json_dict = json.load(fj)
			src_title = src_json_dict.get("title","")

			f_euclid_index.write(str(e_index[0]) + " " + src_title.encode("utf-8") + ":" + str(e_index[1]) + "\n")
			if k+1 == topn:
				break

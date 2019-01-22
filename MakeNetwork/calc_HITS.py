#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import configparser
import codecs
from distutils.util import strtobool

import cvt_to_nxtype2

import sys
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx

def status_writer(dst_dir,opt,comment=None):
	status_file_path = os.path.join(dst_dir,"status.txt")
	with open(status_file_path,"w") as fo:
		for k,v in opt.items():
			print >> fo,k,"=",v
		print >> fo,""
		if comment != None:
			print >> fo,"comment"
			print >> fo,comment

"""ノードの属性を削除"""
def remove_node_attribute(G,name):
	for i in G.node:
		if G.node[i].has_key(name):
			del G.node[i][name]
	return G

def calc_hits_from_redefined_G(search_word,src_pkl_name,exp_name,root_dir,use_to_link="childs",use_from_link=None):
	"""関連フォルダの存在確認"""
	if not os.path.exists(root_dir):
		print "root_dir",root_dir,"is not exist"
		exit()

	exp_dir = os.path.join(root_dir,exp_name)
	if not os.path.exists(exp_dir):
		print "exp_dir",exp_dir,"is not exist"
		exit()

	nx_dir = os.path.join(exp_dir,"nx_datas")
	if not os.path.exists(nx_dir):
		print "nx_dir",exp_dir,"is not exist"
		exit()

	src_pages_dir = os.path.join(root_dir,"Pages")
	if not os.path.exists(src_pages_dir):
		print "src_pages_dir",src_pages_dir,"is not exist"
		exit()

	"""データの読み込み"""
	with open(os.path.join(nx_dir,src_pkl_name),"r") as fi:
		G = pickle.load(fi)

	"""既にある値の削除"""
	G = remove_node_attribute(G,"a_score")
	G = remove_node_attribute(G,"h_score")

	"""計算用のGを作成"""
	calc_G = cvt_to_nxtype2.cvt_jsonfiles_to_G(src_pages_dir,use_to_link=use_to_link,use_from_link=use_from_link,rem_selfloop=True,sel_largest=True)

	h_scores,a_scores = nx.hits(calc_G)
	nx.set_node_attributes(G,"a_score",a_scores)
	nx.set_node_attributes(G,"h_score",h_scores)

	"""データ保存"""
	with open(os.path.join(nx_dir,src_pkl_name),"w") as fo:
		pickle.dump(G,fo)

def main(search_word,src_pkl_name,exp_name,root_dir,weight_key="weight",use_bhits=False):
	"""関連フォルダの存在確認"""
	if not os.path.exists(root_dir):
		print "root_dir",root_dir,"is not exist"
		exit()

	exp_dir = os.path.join(root_dir,exp_name)
	if not os.path.exists(exp_dir):
		print "exp_dir",exp_dir,"is not exist"
		exit()

	nx_dir = os.path.join(exp_dir,"nx_datas")
	if not os.path.exists(nx_dir):
		print "nx_dir",exp_dir,"is not exist"
		exit()

	"""データの読み込み"""
	with open(os.path.join(nx_dir,src_pkl_name),"r") as fi:
		G = pickle.load(fi)

	if use_bhits is True:
		h_scores,a_scores = nx.bhits(G,weight_key=weight_key)
		with open(os.path.join(root_dir,"Progress.txt"),"a") as fo:
			print >> fo,"-----calc_HITS.py-----"
			print >> fo,"この時点で、ノードには代表トピックとその色、from_hosts、to_hosts、オーソリティ・ハブスコア、エッジには重みの情報を渡している。"
			print >> fo,"ノード数：" + str(len(G.node.keys())) + "（G_with_params_comp4_2.gpkl）"
			print >> fo,"weight_key:" + weight_key
			print >> fo,"use_bhits:" + str(use_bhits)
	else:
		h_scores,a_scores = nx.hits(G,weight_key=weight_key)
		with open(os.path.join(root_dir,"Progress.txt"),"a") as fo:
			print >> fo,"-----calc_HITS.py-----"
			print >> fo,"この時点で、ノードには代表トピックとその色、from_hosts、to_hosts、オーソリティ・ハブスコア、エッジには重みの情報を渡している。"
			print >> fo,"ノード数：" + str(len(G.node.keys())) + "（G_with_params_comp4_2.gpkl）"
			print >> fo,"weight_key:" + weight_key
			print >> fo,"use_bhits:" + str(use_bhits)

	nx.set_node_attributes(G,"a_score",a_scores)
	nx.set_node_attributes(G,"h_score",h_scores)

	"""データ保存"""
	with open(os.path.join(nx_dir,src_pkl_name),"w") as fo:
		pickle.dump(G,fo)

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
	G_name = "G" + suffix_generator(target=target,is_largest=is_largest)

	flag = inifile.get('hits','flag')
	if flag == "lda":
		comp_func_name = inifile.get('nx','comp_func_name')
		K = int(inifile.get('lda','K'))
		exp_name = "K" + unicode(K) + suffix_generator(target=target,is_largest=is_largest)
	elif flag == "d2v":
		# comp_func_name = "cos_sim"
		comp_func_name = "euclid"
		D = int(inifile.get('d2v','size'))
		exp_name = "D" + unicode(D) + suffix_generator(target=target,is_largest=is_largest)

	src_pkl_name = "G_with_params_" + comp_func_name + ".gpkl"

	weight_key = inifile.get('hits','weight_key')
	use_bhits = strtobool(inifile.get('hits','use_bhits'))

	main(search_word,src_pkl_name,exp_name,root_dir,weight_key=weight_key,use_bhits=use_bhits)
	#use_to_link="childs"
	#calc_hits_from_redefined_G(search_word,src_pkl_name,exp_name,root_dir,use_to_link=use_to_link)

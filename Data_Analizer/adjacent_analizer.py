#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特定のノードに対して，その隣接ノードの特性を解析する．
リンク元とリンク先それぞれについて解析．
リンク元・リンク先のトピック分布の割合を取得．
"""

import os
import cPickle as pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
import configparser
import codecs
from distutils.util import strtobool

import sys
sys.path.append("../MyPythonModule")
from LDA_kai import LDA
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")

COLORLIST_R = [r"#EB6100",r"#F39800",r"#FCC800",r"#FFF100",r"#CFDB00",r"#8FC31F",r"#22AC38",r"#009944",r"#009B6B",r"#009E96",r"#00A0C1",r"#00A0E9",r"#0086D1",r"#0068B7",r"#00479D",r"#1D2088",r"#601986",r"#920783",r"#BE0081",r"#E4007F",r"#E5006A",r"#E5004F",r"#E60033"]
COLORLIST = [c for c in COLORLIST_R[::2]]#色のステップ調整

"""
リンク先・リンク元のノード番号を取得
@attr
G:Graph of networkx
node_no:the number of node.collect adjacents around  this node.
link_type:select link direction. in or out.
@ret
list of node numbers
"""
def collect_adjacents(G,node_no,link_type):
	ret_list = []
	if link_type == "in":
		for edge in G.in_edges(node_no):
			ret_list.append(edge[0])
	elif link_type == "out":
		for edge in G.out_edges(node_no):
			ret_list.append(edge[1])
	return ret_list

"""
トピック分布の合計を取得
@attr
lda:instance of LDA_kai
targets:list of page id(file_no)s
@ret
nummpy arrary.1 row and K(topics) cols. there are sumation of targets' topic distribution
"""
def topic_summarizer(lda,targets):
	file_id_dict_inv = {v:k for k, v in lda.file_id_dict.items()}#ファイル名とLDAでの文書番号(逆引き)．LDAの方に作っとけばよかった．．．
	sum_topics = np.zeros(lda.K,dtype=np.float32)
	thetas = lda.theta()

	for i,file_no in enumerate(targets):
		lda_id = file_id_dict_inv[file_no]
		sum_topics += thetas[lda_id]
	return sum_topics

"""
@attr
G:Graph of networkx
lda:instance of LDA_kai
node_no:the number of node.collect adjacents around  this node.
link_type:select link direction. in or out.
@ret
nummpy arrary.1 row and K(topics) cols. there are sumation of target adjacents' topic distribution
"""
def get_adjecents_topics(G,lda,node_no,link_type):
	node_list = collect_adjacents(G,node_no,link_type)
	return topic_summarizer(lda,node_list)

""""make pie chart denotes topic ratio"""
def make_topic_ratio_graph(theta,title="topics"):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	labels = [unicode(x+1) for x in range(len(theta))]
	plt.rcParams['font.size'] = 20.0
	ax.pie(theta,colors=COLORLIST[:len(theta)],labels=labels,startangle=90,radius=0.2, center=(0.5, 0.5), frame=True,counterclock=False)
	ax.axis("off")
	ax.axis('equal')
	fig.canvas.set_window_title(title)
	#ax.set_title(title)
	fig.set_facecolor('w')
	#plt.savefig(os.path.join(exp_dir,"Topic"+unicode(lda.K)+"_share_pie.png"))

def main(params):
	nx_dir = params.get("nx_dir")
	src_pkl_name = params.get("src_pkl_name")
	exp_dir = os.path.join(params["root_dir"],params["exp_name"])
	with open(os.path.join(nx_dir,src_pkl_name),"r") as fi:
		G = pickle.load(fi)
	with open(os.path.join(exp_dir,"instance.pkl"),"r") as fi:
	   lda = pickle.load(fi)

	tgt_node = params["target_node"]

	"""リンク元・リンク先のトピック分布の合計"""
	parent_topics = get_adjecents_topics(G,lda,tgt_node,"in")
	child_topics = get_adjecents_topics(G,lda,tgt_node,"out")

	print "parent_topics",parent_topics
	make_topic_ratio_graph(parent_topics,title="parents")
	print "child_topics",child_topics
	make_topic_ratio_graph(child_topics,title="childlen")
	plt.show()

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

if __name__=="__main__":
	"""設定ファイルの読み込み"""
	inifile = configparser.ConfigParser(allow_no_value = True,interpolation = configparser.ExtendedInterpolation())
	inifile.readfp(codecs.open("./analize.ini",'r','utf8'))

	params = {}
	params["search_word"] = inifile.get('options','search_word')
	params["max_page"] = int(inifile.get('options','max_page'))
	add_childs = strtobool(inifile.get('options','add_childs'))
	append = strtobool(inifile.get('options','append'))
	save_dir = inifile.get('other_settings','save_dir')
	params["root_dir"] = save_dir + suffix_generator_root(params["search_word"],params["max_page"],add_childs,append)

	params["is_largest"] = strtobool(inifile.get('options','is_largest'))
	params["target"] = inifile.get('options','target')
	params["K"] = int(inifile.get('lda','K'))
	params["exp_name"] = "K" + unicode(params["K"]) + suffix_generator(params["target"],params["is_largest"])

	params["comp_func_name"] = inifile.get('nx','comp_func_name')
	params["nx_dir"] = os.path.join(os.path.join(params["root_dir"],params["exp_name"]),"nx_datas")
	params["src_pkl_name"] = "G_with_params_" + params["comp_func_name"] + ".gpkl"

	params["target_node"] = int(inifile.get('target','target_node'))

	main(params)

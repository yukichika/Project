#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import configparser
import codecs
from distutils.util import strtobool

import cvt_to_nxtype2
import LDA_for_SS
import LDA_modify_for_graph
import calc_HITS
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

"""LDAの条件を保存"""
def status_writer(dst_dir,opt,comment=None):
	status_file_path = os.path.join(dst_dir,"status.txt")
	with open(status_file_path,"w") as fo:
		for k,v in opt.items():
			print >> fo,k,"=",v
		print >> fo,""
		if comment != None:
			print >> fo,"comment"
			print >> fo,comment

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

	"""収集したリンク情報をnx形式に変換"""
	is_largest = strtobool(inifile.get('options','is_largest'))
	target = inifile.get('options','target')
	G_name = "G" + suffix_generator(target=target,is_largest=is_largest)
	use_to_link = inifile.get('options','use_to_link')

	print("-----収集したリンク情報をnx形式に変換-----")
	cvt_to_nxtype2.main(root_dir=root_dir,sel_largest=is_largest,G_name=G_name,rem_selfloop=True,use_to_link=use_to_link)


	"""Subclass_Summarizer用のLDA実行"""
	"""LDAパラメータの設定"""
	K = int(inifile.get('lda','K'))
	iteration = int(inifile.get('lda','iteration'))
	alpha = float(inifile.get('lda','alpha'))
	beta = float(inifile.get('lda','beta'))
	no_below = int(inifile.get('lda','no_below'))
	no_above = float(inifile.get('lda','no_above'))
	no_less = int(inifile.get('lda','no_less'))
	do_hparam_update = strtobool(inifile.get('lda','do_hparam_update'))

	"""収集したwebページのうち，実際に使用する対象のリスト．（リンクを持っていないものなどを省く）"""
	file_id_list = []
	if is_largest == True:
		with open(os.path.join(root_dir,"file_id_list.list")) as fi:
		   file_id_list = pickle.load(fi)

	"""jsonからchasenへ"""
	chasen_dir_name = "Chasen" + suffix_generator(target,is_largest)
	print("-----jsonからchasenへ-----")
	LDA_for_SS.make_chasens(root_dir,target=target,chasen_dir_name=chasen_dir_name,target_list=file_id_list)

	"""chasenからLDA実行"""
	exp_name = "K" + unicode(K) + suffix_generator(target,is_largest)
	comment = inifile.get('lda','comment')
	if comment == "":
		comment = None

	try:#2回目以降返値が返ってこないのでエラーになる.
		print("-----chasenからLDA実行-----")
		M,V,doclen_ave = LDA_for_SS.main(root_dir=root_dir,K=K,iteration=iteration,smartinit=True,no_below=no_below,no_above=no_above,no_less=no_less,alpha=alpha,beta=beta,target_list=file_id_list,chasen_dir_name=chasen_dir_name,exp_name=exp_name,do_hparam_update=do_hparam_update)
		status_writer(os.path.join(root_dir,exp_name),{"topic_num":K,"M":M,"V":V,"doclen_ave":doclen_ave,"iteration":iteration,"alpha":alpha,"beta":beta,"no_below":no_below,"no_above":no_above,"no_less":no_less,"is_largest":is_largest,"do_hparam_update":do_hparam_update},comment=comment)
	except:
		pass

	"""LDA結果を重みとしてnxグラフに反映"""
	comp_func_name = inifile.get('nx','comp_func_name')
	void_node_remove = strtobool(inifile.get('nx','void_node_remove'))
	print("-----LDAの結果を重みとしてnxグラフに反映-----")
	LDA_modify_for_graph.main(root_dir=root_dir,exp_name=exp_name,comp_func_name=comp_func_name,G_name=G_name,void_node_remove=void_node_remove,is_largest=is_largest)#is_largestはremoveする際に効く

	"""
	arrange_G_data.py　=>　calc_HITS.py（ここまでが可視化前の処理）
	可視化はInteractive_Graph_Visualizer_Qt.pyを用いる．
	"""

	"""
	nxグラフを可視化
	"""
	# nx_dir = os.path.join(os.path.join(root_dir,exp_name),"nx_datas")
	# src_pkl_name = "G_with_params_" + comp_func_name + ".gpkl"
	# weights_pkl_name = "all_node_weights_" + comp_func_name + ".gpkl"
    #
	# draw_option={
	# 	"comp_func_name":comp_func_name,#距離の計算方法
	# 	#"weight_type":[],
	# 	"weight_type":["ATTR","REPUL"],
	# 	#"weight_type":["ATTR","REPUL","HITS"],#オーソリティかハブかはsize_attrで指定
	# 	"node_type":"COMP1",
	# 	# "node_type":"REPR",
	# 	#"node_type":"PIE",
	# 	"do_rescale":True,
	# 	"with_label":False,
	# 	#"size_attr":"a_score",
	# 	#"size_attr":0.02,
	# 	"size_attr":2000,
	# 	"lumine":200,
	# 	"cmap":"jet",
	# 	#"color_map_by":"phi"
	# 	"color_map_by":"theta"
	# 	#"color_map_by":"pie"
	# 	#"color_map_by":None
	# 	}
	# # make_network_by_nx.main(search_word=search_word,src_pkl_name=src_pkl_name,weights_pkl_name=weights_pkl_name,exp_name=exp_name,root_dir=root_dir,nx_dir=nx_dir,topics_K=K,draw_option=draw_option)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import json
import xlsxwriter
import numpy as np
import glob
from sklearn import decomposition
import matplotlib.cm as cm
import configparser
import codecs
from distutils.util import strtobool

import sys
sys.path.append("../MyPythonModule")
from LDA_kai import LDA
sys.path.append("../Interactive_Graph_Visualizer/Interactive_Graph_Visualizer")
import LDA_PCA
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")
import networkx as nx

def cvtRGBAflt2HTML(rgba):
	if isinstance(rgba, tuple):
		rgba = np.array(rgba)
	rgb = rgba[:3]
	rgb_uint = (rgb*255).astype(np.uint8)
	return LDA_PCA.cvtRGB_to_HTML(rgb_uint)

col_conv = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def convert_to_excelpos(row,col):
	return col_conv[col] + unicode(row+1)

def	create_file_analize_sheet(book,src_pages_dir,exp_dir,lda,d2v,tgt_params,pie_dir=None,G_path=None):
	sheet = book.add_worksheet("collection analize")

	draw_topics_flag = False
	if "topics" in tgt_params:
		draw_topics_flag = True
		tgt_params.remove("topics")

	if "hits" in tgt_params:
		tgt_params.remove("hits")
		tgt_params.append("auth_score")
		tgt_params.append("hub_score")
		with open(os.path.join(root_dir,G_path)) as fi:
			G = pickle.load(fi)

	if "pca_lda" in tgt_params:
		theta = lda.theta()[:len(lda.docs)]
		pca = decomposition.PCA(1)
		pca.fit(theta)
		theta_pca = pca.transform(theta)
		reg_theta_pca = (theta_pca-theta_pca.min())/(theta_pca.max()-theta_pca.min())#0~1に正規化
		# cmap = cm.jet_r
		cmap = cm.jet

	if "pca_d2v" in tgt_params:
		vecs = [d2v[x] for x in d2v.keys()]
		pca = decomposition.PCA(1)
		pca.fit(vecs)
		vecs_pca = pca.transform(vecs)
		reg_vecs_pca = (vecs_pca-vecs_pca.min())/(vecs_pca.max()-vecs_pca.min())#0~1に正規化
		# cmap = cm.jet_r
		cmap = cm.jet

	"""1行目（項目名)の追加"""
	for i,param in enumerate(tgt_params):
		sheet.write(0,i,param)
	if draw_topics_flag is True:
		last_col = len(tgt_params)
		for i in range(lda.K):
			sheet.write(0,last_col+i+1,"Topic"+unicode(i+1))#一つスペースを空け，そこにグラフを挿入する

	file_id_dict_inv = {v:k for k, v in lda.file_id_dict.items()}#ファイル名とLDAでの文書番号(逆引き)．LDAの方に作っとけばよかった．．．
	theta = lda.theta()
	for i,file_no in enumerate(G.node.keys()):#全ての除去工程を経た結果がGに入っているため，ここから逆引きするほうが楽
		"""ファイル番号を取得してjson取得"""
		with open(os.path.join(src_pages_dir,unicode(file_no)+".json"),"r") as fj:
			node = json.load(fj)
		id = file_id_dict_inv[file_no]
		tgt_row = i+1

		for j,param in enumerate(tgt_params):
			val = 0
			c_format = None
			if param == "id":
				val = id
			elif param == "name_id":
				val = file_no
			elif param == "domain":
				url = node.get("url")
				val = url.split("/")[2]
			elif param == "len(text)":
				if(node.get("text") != None):
					val = len(node.get("text"))
			elif param == "repTopic":
				val = int(lda.n_m_z[id].argmax()+1)
			elif param == "len_parents":
				parents = node.get("parents")
				if parents != None:
					val = len(parents)
			elif param == "len_childs":
				childs = node.get("childs")
				if childs != None:
					val = len(childs)
			elif param == "to_int_links":
				to_int_links = node.get("to_int_links")
				if to_int_links != None:
					val = len(to_int_links)
			elif param == "to_ext_links":
				to_ext_links = node.get("to_ext_links")
				if to_ext_links != None:
					val = len(to_ext_links)
			elif param == "auth_score":
				val = G.node.get(file_no).get("a_score",-1)
			elif param == "hub_score":
				val = G.node.get(file_no).get("h_score",-1)
			elif param == "pie":
				sheet.insert_image(tgt_row,j,os.path.join(pie_dir,unicode(lda.file_id_dict[id])+".png"))
				continue
			elif param == "pca_lda":
				val = float(reg_theta_pca[id])
				c_format = book.add_format()
				#c_format.set_pattern(1)
				c_format.set_bg_color(cvtRGBAflt2HTML(cmap(val)))
			elif param == "pca_d2v":
				val = float(reg_vecs_pca[i])
				c_format = book.add_format()
				#c_format.set_pattern(1)
				c_format.set_bg_color(cvtRGBAflt2HTML(cmap(val)))

			else:
				val = node.get(param)

			sheet.write(tgt_row,j,val,c_format)

def main(root_dir,expname,newexpname,tgt_params,G_name=None,**kwargs):
	"""関連フォルダの存在確認"""
	if not os.path.exists(root_dir):
		print "root_dir",root_dir,"is not exist"
		exit()

	exp_dir = os.path.join(root_dir,expname)
	if not os.path.exists(exp_dir):
		print "exp_dir",exp_dir,"is not exist"
		exit()

	new_exp_dir = os.path.join(root_dir,newexpname)
	if not os.path.exists(exp_dir):
		print "new_exp_dir",new_exp_dir,"is not exist"
		exit()

	src_pages_dir = os.path.join(root_dir,"Pages")
	if "topics" in tgt_params:
		if not os.path.exists(src_pages_dir):
			print "pages_dir",src_pages_dir,"is not exist"
			exit()

	pie_dir = None
	if "pie" in tgt_params:
		pie_dir = os.path.join(exp_dir,"pie_graphs")
		if not os.path.exists(pie_dir):
			print "pie_dir",pie_dir,"is not exist"
			exit()

	G_path = None
	if "hits" in tgt_params:
		nx_dir = os.path.join(exp_dir,"nx_datas")
		G_path = glob.glob(nx_dir+"/G*gpkl")[0]
		if not os.path.exists(G_path):
			print "G",G_path,"is not exist"
			exit()

	save_name = kwargs.get("save_name","collection_datas.xlsx")
	top_n = kwargs.get("top_n",20)

	"""ファイルの読み込み"""
	with open(os.path.join(exp_dir,"instance.pkl")) as fi:
	   lda = pickle.load(fi)
	with open(os.path.join(new_exp_dir,"doc2vec.pkl")) as fi:
	   d2v = pickle.load(fi)

	book = xlsxwriter.Workbook(os.path.join(new_exp_dir,save_name))
	"""Webページの情報"""
	create_file_analize_sheet(book,src_pages_dir,exp_dir,lda,d2v,tgt_params,pie_dir=pie_dir,G_path=G_path)

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
	size = int(inifile.get('d2v','size'))
	new_exp_name = "D" + unicode(size) + suffix_generator(target,is_largest)

	save_name = "collection_datas_d2v.xlsx"
	top_n = 20

	tgt_params = [
		"id",#LDAの番号
		"name_id",#ファイル名の番号
		"title",#タイトル
		"len(text)",#文字数（単語数ではない）
		"url",#URL
		"domain",#ドメイン名
		"len_parents",#["parents"]の数
		"len_childs",#["childs"]の数
		"to_int_links",#["to_int_links"]の数
		"to_ext_links",#["to_ext_links"]の数
		"repTopic",#代表トピック
		"hits",#HITSスコア
		"pca_lda",#LDAの主成分分析
		"pca_d2v"#D2Vの主成分分析
		]

	main(root_dir=root_dir,expname=exp_name,newexpname=new_exp_name,tgt_params=tgt_params,save_name=save_name,top_n=top_n)

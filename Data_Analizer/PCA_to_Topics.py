#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
やりたいこと
PCAで次元圧縮した空間と元のトピック分布との対応づけ
トピック空間での主成分ベクトルを用いてカラーマップから単語分布を逆引きする．
"""

import numpy as np
import os
import os.path
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import matplotlib.font_manager
from sklearn import decomposition
import xlsxwriter
from sklearn.feature_extraction.text import TfidfVectorizer
import configparser
import codecs
from distutils.util import strtobool

import sys
sys.path.append("../MyPythonModule")
from LDA_kai import LDA
sys.path.append("../Interactive_Graph_Visualizer/Interactive_Graph_Visualizer")
import LDA_PCA

"""0~1に正規化した値をstart_angleだけずらして0~2piに"""
def circler_color_converter(values,start_angle):
	values = values*2*np.pi#0~1⇒0~2piに
	values = values+start_angle*np.pi
	np.where(values<2*np.pi,values,values-2*np.pi)
	return values

"""上記関数の逆変換.jetの場合は不要"""
def circler_color_deconverter(values,start_angle):
	values = values-start_angle*np.pi
	np.where(values>=0,values,values+2*np.pi)
	values = values/(2*np.pi)
	return values

def calc_composite_thetas(lda,pca,theta_pca,resolution=10,output=False):
	#steps = np.linspace(0, 1, 50)
	steps = np.linspace(theta_pca.min(), theta_pca.max(), resolution)
	#theta = lda.theta()[:len(lda.docs)]

	#molecs = pca.inverse_transform(theta_pca)-pca.inverse_transform(theta_pca).min(axis=1)[np.newaxis].T+lda.alpha
	#rev_theta = molecs/molecs.sum(axis=1)[np.newaxis].T

	"""
	主成分分析結果から元のトピック分布を再計算
	理論的には範囲内に収まるはずだが，なぜか負の値を持つことがあるため0~1に収まるように正規化
	"""
	#molecs = pca.inverse_transform(steps[np.newaxis].T)-pca.inverse_transform(steps[np.newaxis].T).min(axis=1)[np.newaxis].T+lda.alpha

	"""負の数を0にするパターン"""
	molecs = pca.inverse_transform(steps[np.newaxis].T)
	molecs = np.where(molecs>0,molecs,0)

	rev_thetas = molecs/molecs.sum(axis=1)[np.newaxis].T

	if output == True:
		with open("rev_thetas.csv","w") as fo:
			for rev_theta in rev_thetas:
				for val in rev_theta:
					print >> fo,val,
				print >> fo,""

	return rev_thetas

def output_worddist_lda(lda,word_dists,dir_path=".",topn=20):
	output_topn(lda.vocas,word_dists,dir_path=dir_path,topn=topn)

def cvtRGBAflt2HTML(rgba):
	if isinstance(rgba, tuple):
		rgba = np.array(rgba)
	rgb = rgba[:3]
	rgb_uint = (rgb*255).astype(np.uint8)
	return LDA_PCA.cvtRGB_to_HTML(rgb_uint)

def output_worddist_tfidf(lda,word_dists,dir_path=".",topn=20):
	uni_docs = []
	for word_dist in word_dists:
		uni_words = u""
		for word_id in np.argsort(-word_dist)[:100]:
			uni_words += lda.vocas[word_id].decode("utf8")#.encode("sjis")
			uni_words += u" "
		uni_docs.append(uni_words)

	vectorizer = TfidfVectorizer(use_idf=True,lowercase=False)
	tfidf = vectorizer.fit(uni_docs)
	features = tfidf.transform(uni_docs).toarray()
	terms = tfidf.get_feature_names()

	output_topn(terms,features,dir_path=dir_path,topn=topn)

def output_worddist_lda_tfidf(lda,word_dists,dir_path=".",topn=20):
	uni_docs = []
	org_word_to_plb_list = []#ユニコード単語をキーに確率を引く辞書のリスト．後で復元する際に使う．
	for word_dist in word_dists:
		org_word_to_plb = {}
		uni_words = u""
		for word_id in np.argsort(-word_dist)[:100]:
			uni_words += lda.vocas[word_id].decode("utf8")#.encode("sjis")
			uni_words += u" "
			org_word_to_plb[lda.vocas[word_id].decode("utf8")]=word_dist[word_id]
		uni_docs.append(uni_words)
		org_word_to_plb_list.append(org_word_to_plb)

	vectorizer = TfidfVectorizer(use_idf=True,lowercase=False)
	tfidf = vectorizer.fit(uni_docs)
	features = tfidf.transform(uni_docs).toarray()
	terms = tfidf.get_feature_names()

	for feature,org_word_to_plb_ in zip(features,org_word_to_plb_list):
		for i,term in enumerate(terms):
			if feature[i] == 0.:
				continue
			feature[i] = feature[i]*org_word_to_plb_[term]

	output_topn(terms,features,dir_path=dir_path,topn=topn)

def output_topn(terms,features,dir_path=".",topn=20):
	c_map = cm.jet_r#環境によってPCAの値が反転する？ため，カラーマップを反転させて対応
	# c_map = cm.jet#環境によってPCAの値が反転する？ため，カラーマップを反転させて対応
	steps = np.linspace(0., 1., len(features))
	book = xlsxwriter.Workbook(os.path.join(dir_path,"words.xlsx"))
	sheet = book.add_worksheet("words")
	bold = book.add_format({'bold': True})
	for col,feature in enumerate(features):
		for row,word_id in enumerate(np.argsort(-feature)[:topn]):
			flag_cnt = 0
			if col == 0:
				flag_cnt += 1
			else:
				pre_rank = np.argsort(-features[col-1])[:100]
				pre_rank = np.where(pre_rank==word_id)[0]
				if len(pre_rank) == 0 or pre_rank[0] > row:
					flag_cnt += 1
			if col == (len(features)-1):
				flag_cnt += 1
			else:
				nxt_rank = np.argsort(-features[col+1])[:100]
				nxt_rank = np.where(nxt_rank==word_id)[0]
				if len(nxt_rank) == 0 or nxt_rank[0] > row:
					flag_cnt += 1

			str_word = terms[word_id]
			# str_word = terms[word_id].decode("utf-8")
			# str_word = terms[word_id].encode("utf8")
			if flag_cnt >= 2:
				sheet.write(row,col,str_word,bold)
			else:
				sheet.write(row,col,str_word)
		c_format = book.add_format()
		c_format.set_pattern(1)
		c_format.set_bg_color(cvtRGBAflt2HTML(c_map(steps[col])))
		sheet.write(row+1,col,"",c_format)
	book.close()

def calc_composite_worddist(lda,comp_type="COMP1",lumine=255,cmap="lch",output_option={}):
	calc_method = output_option.get("calc_method",output_worddist_lda)
	topn = output_option.get("topn",50)

	"""thetaの方を主成分分析で1次元にして彩色"""
	theta = lda.theta()[:len(lda.docs)]
	pca = decomposition.PCA(1)
	pca.fit(theta)
	theta_pca = pca.transform(theta)
	theta_pca = pca.transform(pca.inverse_transform(theta_pca))#1回の投影ではなぜか値がずれるため再投影...してみたが，特に変わらなかった
	#reg_theta_pca=(theta_pca-theta_pca.min())/(theta_pca.max()-theta_pca.min())#0~1に正規化

	resolution = 10
	rev_thetas = calc_composite_thetas(lda,pca,theta_pca,resolution=resolution,output=True)
	word_dists = rev_thetas.dot(lda.phi())

	calc_method(lda,word_dists,topn=topn)

def main(params):
	root_dir = params.get("root_dir")
	exp_name = params.get("exp_name")
	exp_dir = os.path.join(root_dir,exp_name)
	#nx_dir = params.get("nx_dir")
	#src_pkl_name = params.get("src_pkl_name")
	#weights_pkl_name = params.get("weights_pkl_name")

	draw_option = params.get("draw_option")
	lumine = draw_option.get("lumine")
	cmap = draw_option.get("cmap")
	comp_type = draw_option.get("comp_type")

	output_option = params.get("output_option")

	with open(os.path.join(exp_dir,"instance.pkl"),"r") as fi:
	   lda = pickle.load(fi)

	calc_composite_worddist(lda,comp_type,lumine=lumine,cmap=cmap,output_option=output_option)

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
		suffix += "_"+target
	if is_largest == True:
		suffix += "_largest"
	return suffix

if __name__ == "__main__":
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

	draw_option = {"node_type":"COMP1",
				   "lumine":200,
				   "cmap":"jet",}

	params["draw_option"] = draw_option

	"""
	ここからが出力する分布に関する設定
	calc_method:output_worddist_lda
			   :output_worddist_tfidf
			   :output_worddist_lda_tfidf
	"""
	output_option = {"calc_method":output_worddist_lda_tfidf,
				   "topn":20}

	params["output_option"] = output_option

	main(params)
	plt.show()

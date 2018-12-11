#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import configparser
import codecs
from distutils.util import strtobool
from tqdm import tqdm

import sys
sys.path.append("../../Preprocessing")
import Sentence
import Delete
import Wakati

"""保存名の決定"""
def suffix_generator_root(search_word,max_page,add_childs,append):
	suffix = "_" + search_word
	suffix += "_" + str(max_page)
	if add_childs:
		suffix += "_add_childs"
	if append:
		suffix += "_append"
	return suffix

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
	Myexttext_raw = os.path.join(root_dir,"Myexttext_raw")
	Myexttext_pre = os.path.join(root_dir,"Myexttext_pre")
	os.makedirs(Myexttext_pre,exist_ok=True)

	node_list = []
	with open(os.path.join(root_dir,"file_id_list2.list"),'rb') as fi:
	   node_list = pickle.load(fi)
	print("ノード数：" + str(len(node_list)))

	"""前処理"""
	for node in tqdm(node_list):
		with open(os.path.join(Myexttext_raw,str(node) + ".txt"),'r') as fi:
			text = fi.read()
		sentences = Sentence.sentence(text)
		with open(os.path.join(Myexttext_pre,str(node) + ".txt"),'w') as fo:
			for sentence in sentences:
				fo.write(Delete.delete_wikipedia(sentence) + "\n")

	"""ベクトル化"""
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

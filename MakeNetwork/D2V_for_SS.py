#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import configparser
import codecs
from distutils.util import strtobool
from tqdm import tqdm
from gensim import models

import Preprocessing

import sys
sys.path.append("../Crawler")
import Mymodule

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
	Myexttext_raw = os.path.join(root_dir,"Myexttext_raw")
	Myexttext_pre = os.path.join(root_dir,"Myexttext_pre")

	"""前処理"""
	if os.path.exists(Myexttext_pre):
		print("preprocess finished.")
	else:
		print("-----preprocess-----")
		os.mkdir(Myexttext_pre)

		node_list = []
		with open(os.path.join(root_dir,"file_id_list2.list"),'rb') as fi:
			node_list = pickle.load(fi)
		print("ノード数：" + str(len(node_list)))

		for node in tqdm(node_list):
			with open(os.path.join(Myexttext_raw,str(node) + ".txt"),'r') as fi:
				text = fi.read()
				text = Preprocessing.delete_pre(text)
			sentences = Preprocessing.simple_sentence(text)
			with open(os.path.join(Myexttext_pre,str(node) + ".txt"),'w') as fo:
				for j,sentence in enumerate(sentences):
					words = Preprocessing.wakati(sentence).split(" ")
					for i,word in enumerate(words):
						words[i] = Preprocessing.delete_aft(word)
					words = [x for x in words if x]
					text = " ".join(words)
					if not j == len(sentences):
						fo.write(text + " \n")
					else:
						fo.write(text)

	"""ベクトル化"""
	is_largest = strtobool(inifile.get('options','is_largest'))
	target = inifile.get('options','target')
	size = int(inifile.get('d2v','size'))
	exp_name_new = "D" + str(size) + suffix_generator(target,is_largest)
	exp_dir_new = os.path.join(root_dir,exp_name_new)

	if os.path.exists(exp_dir_new):
		print("vectorize finished.")
	else:
		print("-----vectorize-----")
		os.mkdir(exp_dir_new)

		INPUT_MODEL = u"/home/yukichika/ドキュメント/Doc2vec_model/Wikipedia809710_dm_100_w5_m5_20.model"
		model = models.Doc2Vec.load(INPUT_MODEL)

		vectors = {}
		targets = os.listdir(Myexttext_pre)
		Mymodule.sort_nicely(targets)

		for file in tqdm(targets):
			node_no = int(file.split(".")[0])
			with open(os.path.join(Myexttext_pre,file),'r') as fi:
				lines = fi.readlines()
				words = []
				for line in lines:
					words.extend(line.split(" "))
					vector = model.infer_vector(words)
					vectors[node_no] = vector

		with open(os.path.join(exp_dir_new,"doc2vec.pkl"),'wb') as fo:
			pickle.dump(vectors,fo,protocol=2)

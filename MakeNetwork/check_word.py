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

	is_largest = strtobool(inifile.get('options','is_largest'))
	target = inifile.get('options','target')
	size = int(inifile.get('d2v','size'))
	exp_name_new = "D" + str(size) + suffix_generator(target,is_largest)
	exp_dir_new = os.path.join(root_dir,exp_name_new)

	vocab = Preprocessing.words_vocab(Myexttext_pre)
	average = Preprocessing.words_average(Myexttext_pre)
	file_no = len(os.listdir(Myexttext_pre))

	print("ファイル数：" + str(file_no))
	print("語彙数：" + str(vocab))
	print("平均単語数：" + str(average))

	with open(os.path.join(exp_dir_new,"doc2vec.txt"),'w',encoding="utf-8") as fo:
		fo.write("ファイル数：" + str(file_no) + "\n")
		fo.write("語彙数：" + str(vocab) + "\n")
		fo.write("平均単語数：" + str(average) + "\n")

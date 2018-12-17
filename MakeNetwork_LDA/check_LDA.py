#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import configparser
import codecs
import os
from distutils.util import strtobool

import series_act

import sys
sys.path.append("../MyPythonModule")
from LDA_kai import LDA

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
	root_dir = save_dir + series_act.suffix_generator_root(search_word,max_page,add_childs,append)

	is_largest = strtobool(inifile.get('options','is_largest'))
	target = inifile.get('options','target')
	K = int(inifile.get('lda','K'))
	exp_name = "K" + unicode(K) + series_act.suffix_generator(target,is_largest)

	ldafile = os.path.join(os.path.join(root_dir,exp_name),"instance.pkl")
	with open(ldafile,'r') as fi:
		lda = pickle.load(fi)

	print("語彙数：" + str(len(lda.vocas)))
	for v in lda.vocas:
		print(v)

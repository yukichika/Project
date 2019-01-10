#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from tqdm import tqdm
import re

"""コサイン類似度"""
def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

"""ユークリッド距離"""
def euclid(p,q):
	weight = np.sqrt(np.power(p-q,2).sum())
	return weight

"""指数で正規化したユークリッド距離"""
def compare4_2(p,q):
	weight = np.exp(-((p-q)**2).sum())
	return weight

if __name__ == "__main__":
	INPUT = u"/home/yukichika/ドキュメント/Doc2vec_vector"
	with open(os.path.join(INPUT,"Wikipedia809710_to_Wiki(202445).pkl"),'rb') as fi:
		doc2vec_vectors = pickle.load(fi)

	topn = 50
	pos1 = "日本酒(8640).txt"
	neg1 = "刺身(15400).txt"
	pos2 = "チーズ(4584).txt"
	target_name = pos1.split(".")[0] + "-" + neg1.split(".")[0] + "+" + pos2.split(".")[0]
	target_name = re.sub(r'\(.*?\)',"",target_name)
	target = pos1 + "-" + neg1 + "+" + pos2

	print("記事数：" + str(len(doc2vec_vectors)))
	print("対象：" + target)

	cos = []#コサイン類似度
	euclid_ = []#ユークリッド距離
	euclid_index = []#指数で正規化したユークリッド距離

	p_dst = doc2vec_vectors[pos1] - doc2vec_vectors[neg1] + doc2vec_vectors[pos2]
	for key,value in tqdm(doc2vec_vectors.items()):
		q_dst = doc2vec_vectors[key]
		weight_cos = cos_sim(p_dst,q_dst)
		weight_euclid = euclid(p_dst,q_dst)
		weight_index = compare4_2(p_dst,q_dst)

		cos.append([key,weight_cos])
		euclid_.append([key,weight_euclid])
		euclid_index.append([key,weight_index])

	cos = sorted(cos,key=lambda x: x[1],reverse=True)
	euclid_ = sorted(euclid_,key=lambda x: x[1],reverse=False)
	euclid_index = sorted(euclid_index,key=lambda x: x[1],reverse=True)


	with open(os.path.join(INPUT,target_name + "_cos.txt"),'w',encoding='UTF-8') as f_cos:
		f_cos.write(target + "\n\n")
		for i,c in enumerate(cos):
			f_cos.write(c[0] + ":" + str(c[1]) + "\n")
			if i+1 == topn:
				break

	with open(os.path.join(INPUT,target_name + "_euclid.txt"),'w',encoding='UTF-8') as f_euclid:
		f_euclid.write(target + "\n\n")
		for j,e in enumerate(euclid_):
			f_euclid.write(e[0] + ":" + str(e[1]) + "\n")
			if j+1 == topn:
				break

	with open(os.path.join(INPUT,target_name + "_euclid_indec.txt"),'w',encoding='UTF-8') as f_euclid_index:
		f_euclid_index.write(target + "\n\n")
		for k,e_index in enumerate(euclid_index):
			f_euclid_index.write(e_index[0] + ":" + str(e_index[1]) + "\n")
			if k+1 == topn:
				break

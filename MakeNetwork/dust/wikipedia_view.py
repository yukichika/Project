#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import decomposition
import codecs
import matplotlib.pyplot as plt

if __name__ == "__main__":
	target1 = "country"
	target2 = "alcohol_"
	INPUT = u"/home/yukichika/ドキュメント/Doc2vec_vector"
	with open(os.path.join(INPUT,"Wikipedia809710_to_Wiki(202445).pkl"),'rb') as fi:
		doc2vec_vectors = pickle.load(fi)

	all_vector = []
	for k,v in tqdm(doc2vec_vectors.items()):
		all_vector.append(v)

	print("全データ数：" + str(len(all_vector)))

	nodes1 = []
	nodes2 = []
	with open(os.path.join(INPUT,"view_target.txt"),'r',encoding='UTF-8') as fi:
		lines = fi.readlines()
		for line in lines:
			line = line.replace("\n","")
			target = line.split("\t")
			nodes1.append(target[0])
			nodes2.append(target[1])

	nodes1_v = []
	nodes2_v = []
	for node1,node2 in zip(nodes1,nodes2):
		nodes1_v.append(doc2vec_vectors[node1])
		nodes2_v.append(doc2vec_vectors[node2])
	total = nodes1_v + nodes2_v

	pca = decomposition.PCA(2)
	pca.fit(total)

	# pca = decomposition.PCA(2)
	# pca.fit(all_vector)

	data_pca1 = pca.transform(nodes1_v)
	data_pca2 = pca.transform(nodes2_v)

	print("node1の数:" + str(len(data_pca1)))
	print("node2の数:" + str(len(data_pca2)))
	print("total:" + str(len(total)))

	x1 = []
	y1 = []
	x2 = []
	y2 = []
	for pca1,pca2 in zip(data_pca1,data_pca2):
		x1.append(pca1[0])
		y1.append(pca1[1])
		x2.append(pca2[0])
		y2.append(pca2[1])

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x1,y1, c='red',label=target1)
	ax.scatter(x2,y2, c='blue',label=target2)

	# ax.set_title('second scatter plot')
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	ax.legend(loc='upper left')
	fig.show()

	plt.savefig(os.path.join(INPUT,target1 + "&" + target2 + ".png"))

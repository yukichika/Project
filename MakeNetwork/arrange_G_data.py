#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
networkxのGデータに後から属性を追加するスクリプト．
土壇場の付け焼刃コードなので参照非推奨
"""

import os
import json
import codecs
import cPickle as pickle
from collections import Counter

import sys
sys.path.append("../MyPythonModule")
import mymodule
sys.path.append("../Interactive_Graph_Visualizer/networkx-master")

"""
urlからドメイン部分を抽出して返す．
FQDNのドットで区切られたブロック数が3以下の場合，ドメイン名はFQDN自身
4以上の場合，下3ブロックをドメインとして返す．
@arg
[IN]url:ドメインを取得したいURL
@ret
(unicode) domain
"""
def domain_detect(url):
	FQDN = url.split("/")[2]
	FQDN_list = FQDN.split(".")
	return u".".join(FQDN_list[-3:])#開始点がリスト長より長くても問題なく無く動く

def main(params):
	root_dir = params.get("root_dir")
	"""関連フォルダの存在確認"""
	if not os.path.exists(root_dir):
		print "root_dir",root_dir,"is not exist"
		exit()

	src_pages_dir = os.path.join(root_dir,"Pages")
	if not os.path.exists(src_pages_dir):
		print "pages_dir",src_pages_dir,"is not exist"
		exit()

	nx_dir = params.get("nx_dir")
	src_pkl_name = params.get("src_pkl_name")
	src_gpkl_path = os.path.join(nx_dir,src_pkl_name)

	with open(src_gpkl_path,"r") as fi:
		G = pickle.load(fi)

	"""各nodeのjson情報を先に格納しておく"""
	node_datas = dict()
	for node in G:
		with open(os.path.join(src_pages_dir,str(node)+".json"),"r") as fj:
			node_data = json.load(fj)
		node_datas[node] = node_data#jsonファイル内の情報の辞書

	for node in G:
		node_data = node_datas[node]
		from_hosts = []
		to_hosts = []
		for in_edge in G.in_edges(node):#入ってくるエッジ
			from_node = in_edge[0]
			fromnode_data = node_datas[from_node]
			from_hosts.append(domain_detect(fromnode_data["url"]))
		for out_edge in G.out_edges(node):#出て行くエッジ
			to_node = out_edge[1]
			tonode_data = node_datas[to_node]
			to_hosts.append(domain_detect(tonode_data["url"]))
		G.node[node]["from_hosts"] = dict(Counter(from_hosts))#Counterインスタンスを辞書型でキャスト
		G.node[node]["to_hosts"] = dict(Counter(to_hosts))

	with open(src_gpkl_path,"w") as fo:
		pickle.dump(G,fo)
	with open(os.path.join(root_dir,"Progress.txt"),"a") as fo:
		print >> fo,"-----arrange_G_data.py-----"
		print >> fo,"この時点で、ノードには代表トピックとその色、from_hosts、to_hostsをエッジには重みの情報を渡している。"
		print >> fo,"from_host=>リンクされているWebページのドメインとその件数の辞書"
		print >> fo,"to_hosts=>リンクしているWebページのドメインとその件数の辞書"
		print >> fo,"ノード数：" + str(len(G.node.keys())) + "（G_with_params_comp4_2.gpkl）"

def del_keys(params):
	root_dir = params.get("root_dir")
	"""関連フォルダの存在確認"""
	if not os.path.exists(root_dir):
		print "root_dir",root_dir,"is not exist"
		exit()

	src_pages_dir = os.path.join(root_dir,"pages")
	if not os.path.exists(src_pages_dir):
		print "pages_dir",src_pages_dir,"is not exist"
		exit()

	nx_dir = params.get("nx_dir")
	src_pkl_name = params.get("src_pkl_name")
	src_gpkl_path = os.path.join(nx_dir,src_pkl_name)
	with open(src_gpkl_path,"r") as fi:
		G = pickle.load(fi)
	for node in G:
		G[node].pop("to_hosts")
		G[node].pop("from_hosts")
	with open(src_gpkl_path,"w") as fo:
		pickle.dump(G,fo)

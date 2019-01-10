#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
やりたいこと
UI付きのグラフ可視化．
静的なUIのみ．
実装予定
・スクロールによるズーム
・ドラッグによる移動
・マウスホバーでタイトル
・マウスホバーで接続先ハイライト
・クリックで詳細表示
・表示ノードの切り替え
・表示リンクの切り替え
"""

import numpy as np
import os
import os.path
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from math import modf#整数と小数の分離
import matplotlib.font_manager
import cv2
from sklearn import decomposition
import json
import codecs

import color_changer
import LDA_PCA
import make_lch_picker

import sys
sys.path.append("../../MyPythonModule")
import mymodule
from LDA_kai import LDA
sys.path.append("../networkx-master")
import networkx as nx

prop = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/meiryo/meiryo.ttc')#pyplotに日本語を使うために必要

"""オプションを読みやすい形式で保存.前処理をしてから渡す"""
def save_drawoption(param_dict,path):
	mymodule.save_option(param_dict,path)

"""pathの位置に乱数があればそれを，無ければ新たに作る"""
def pos_initializer(G,path):
	#pathが存在した場合
	if os.path.exists(path):
		with open(path) as fi:
			pos = pickle.load(fi)
		return pos
	#pathが存在しない場合
	pos = dict()
	for a, d in G.nodes(data=True):
		pos[a] = np.random.rand(2)
	with open(path,"w") as fo:
		pickle.dump(pos,fo)
	return pos

"""
HITSのパラメータに応じてノードのサイズを決定
@ret:{ノード番号:size}の辞書
"""
def calc_nodesize(G,attr="a_score",weight_key="weight",min_size=1000,max_size=5000,use_bhits=True):
	if type(attr) != str and type(attr) != unicode:
		normal_size = max_size - min_size
		normal_size = attr
		#print "all size uniformed"
		return dict([(node_no,normal_size) for node_no in G.node])

	if attr == "a_score" or attr == "h_score":
		#a_scores,h_scores=nx.hits(G)#引数の順番違い．HCG論文提出時にこっちで出してしまっていた．．．
		if use_bhits is True:
			h_scores,a_scores = nx.bhits(G,weight_key=weight_key)
		else:
			h_scores,a_scores = nx.hits(G,weight_key=weight_key)

		if attr == "a_score":
			use_vals = a_scores
		elif attr == "h_score":
			use_vals = h_scores

	if attr == "in_degree":
		use_vals = dict()
		for g in G:
			use_vals[g] = G.in_degree(g)

	max_val = max(use_vals.values())
	size_dict = dict()
	for node_no,node_attr in G.nodes(data=True):
		val = node_attr.get(attr)#論文提出時はauthorityをhubのmaxで割った
		if val == None:
			size = min_size/2
		else:
			size = (val/max_val)*(max_size-min_size) + min_size
		size_dict[node_no] = size
	return size_dict

def circler_color_converter(values,start_angle):
	values = values+start_angle*np.pi
	np.where(values<2*np.pi,values,values-2*np.pi)#（条件式，True，False）
	return values

def cvtLCH_to_HTML(LCH_1channel):
	lch_img = np.ones((2,2,3),dtype=np.float32)*LCH_1channel
	BGR_img = color_changer.cvtLCH2BGR(lch_img)
	RGB_img = cv2.cvtColor(BGR_img,cv2.COLOR_BGR2RGB)
	RGB_1channel = RGB_img[0,0]
	return cvtRGB_to_HTML(RGB_1channel)

"RGB値=>16進数のカラーコード"
def cvtRGB_to_HTML(RGB_1channel):
	R,G,B = RGB_1channel
	R_str = unicode("%02x"%R)
	G_str = unicode("%02x"%G)
	B_str = unicode("%02x"%B)
	return u"#"+R_str+G_str+B_str

"""rgba(jetカラーマップの値)=>RGB(0~255)=>16進数のカラーコード"""
def cvtRGBAflt2HTML(rgba):
	rgb = rgba[0][:3]
	rgb_uint = (rgb*255).astype(np.uint8)
	return LDA_PCA.cvtRGB_to_HTML(rgb_uint)

reg_d2v_pca = 0
def get_color_map_vector1(G,pos,d2v,comp_type="COMP1",lumine=255,cmap="lch"):
	global reg_d2v_pca
	"""doc2vecのベクトルの方を主成分分析で1次元にして彩色"""
	vector = d2v.values()
	pca = decomposition.PCA(1)
	pca.fit(vector)
	d2v_pca = pca.transform(vector)#第一主成分
	reg_d2v_pca = (d2v_pca-d2v_pca.min())/(d2v_pca.max()-d2v_pca.min())#第一主成分を0~1に正規化
	h_values = circler_color_converter(reg_d2v_pca*2*np.pi,0.2).T[0]#列ヴェクトルとして与えられるため，1行に変換
	make_lch_picker.draw_color_hist(h_values,resolution=50,lumine=lumine,color_map=cmap)#色変換の図を表示

	"""寄与率計算のため，再度PCA"""
	pca2 = decomposition.PCA(len(d2v.values()[0]))
	pca2.fit(vector)
	print pca2.explained_variance_ratio_

	if cmap == "lch":
		c_flt = 1.0
		file_id_dict_inv = {v:i for i, v in enumerate(d2v.keys())}
		color_map = {}
		for serial_no,node_no in enumerate(G.node.keys()):
			d2v_no = file_id_dict_inv.get(node_no)
			if d2v_no == None:
				color_map[node_no] = r"#FFFFFF"
				continue
			h_value = h_values[d2v_no]
			lch = np.array((lumine,c_flt,h_value),dtype=np.float32)#（[輝度,？,？]）
			html_color = cvtLCH_to_HTML(lch)
			color_map[node_no] = html_color

	elif cmap == "jet":
		# c_map = cm.jet
		c_map = cm.jet_r#環境によってPCAの値が反転する？ため，カラーマップを反転させて対応
		file_id_dict_inv = {v:i for i, v in enumerate(d2v.keys())}
		color_map = {}
		for serial_no,node_no in enumerate(G.node.keys()):
			d2v_no = file_id_dict_inv.get(node_no)
			if d2v_no == None:
				color_map[node_no] = r"#FFFFFF"
				continue
			color_map[node_no] = cvtRGBAflt2HTML(c_map(reg_d2v_pca[d2v_no]))#（R,G,B）=>16進数のカラーコード
	return color_map

# def cvtRGBAflt2HTML_3D(rgba):
# 	rgb = []
# 	rgb.append(rgba[0][0])
# 	rgb.append(rgba[1][1])
# 	rgb.append(rgba[2][2])
# 	rgb = np.array(rgb)
# 	rgb_uint = (rgb*255).astype(np.uint8)
# 	return LDA_PCA.cvtRGB_to_HTML(rgb_uint)
#
# reg_d2v_pca = 0
# def get_color_map_vector3(G,pos,d2v,comp_type="COMP1",lumine=255,cmap="lch"):
# 	global reg_d2v_pca
# 	"""doc2vecのベクトルの方を主成分分析で3次元にして彩色"""
# 	vector = d2v.values()
# 	pca = decomposition.PCA(3)
# 	pca.fit(vector)
# 	d2v_pca = pca.transform(vector)
# 	reg_d2v_pca = (d2v_pca-d2v_pca.min())/(d2v_pca.max()-d2v_pca.min())#0~1に正規化
#
# 	"""寄与率計算のため，再度PCA"""
# 	pca2 = decomposition.PCA(len(d2v.values()[0]))
# 	pca2.fit(vector)
# 	print pca2.explained_variance_ratio_
#
# 	if cmap == "jet":
# 		# c_map = cm.jet
# 		c_map = cm.jet_r#環境によってPCAの値が反転する？ため，カラーマップを反転させて対応
# 		file_id_dict_inv = {v:i for i, v in enumerate(d2v.keys())}
# 		color_map = {}
# 		for serial_no,node_no in enumerate(G.node.keys()):
# 			d2v_no = file_id_dict_inv.get(node_no)
# 			if d2v_no == None:
# 				color_map[node_no] = r"#FFFFFF"
# 				continue
# 			color_map[node_no] = cvtRGBAflt2HTML_3D(c_map(reg_d2v_pca[d2v_no]))
# 	return color_map

def draw_node_with_lch(G,pos,**kwargs):
	d2v = kwargs.get("d2v")
	size = kwargs.get("size")
	draw_option = kwargs.get("draw_option")

	color_map_by = draw_option.get("color_map_by")
	comp_type = draw_option.get("comp_type")
	lumine = draw_option.get("lumine")
	cmap = draw_option.get("cmap")
	ax = draw_option.get("ax")
	pick_func = draw_option.get("pick_func")
	lamb = draw_option.get("lamb")

	if color_map_by == "vector1":#主成分分析の対象がdoc2vecのベクトルで，1次元に主成分分析
		color_map = get_color_map_vector1(G,pos,d2v,comp_type,lumine=lumine,cmap=cmap)
	# elif color_map_by == "vector3":#主成分分析の対象がdoc2vecのベクトルで，3次元に主成分分析
	# 	color_map = get_color_map_vector3(G,pos,d2v,comp_type,lumine=lumine,cmap=cmap)
	elif color_map_by == None:#無色
		color_map = dict.fromkeys(G,"#FFFFFF")

	node_color = color_map.values()
	size_array = size.values()
	node_collection = nx.draw_networkx_nodes(G,pos=pos,node_color=node_color,node_size=size_array,ax=ax,pick_func=pick_func,lamb=lamb)
	return node_collection,color_map

def draw_network(G,pos,**kwargs):
	draw_option = kwargs.get("draw_option")
	node_type = draw_option.get("node_type")
	ax = draw_option.get("ax")
	size = kwargs.get("size")
	pick_func = draw_option.get("pick_func")
	lamb = draw_option.get("lamb")

	color_map = None
	if node_type == "COMP1":#doc2vecのベクトルを主成分分析で可視化
		node_collection,color_map = draw_node_with_lch(G,pos,**kwargs)
	elif node_type == "kmeans100_j":
		k100_dict = nx.get_node_attributes(G,"kmeans_100")
		k100 = np.array(k100_dict.values())
		k100 = k100.astype("float32")
		k100 = (k100-k100.min())/(k100.max()-k100.min())

		c_map = cm.jet_r
		color_map = {}
		for serial_no,node_no in enumerate(G.node.keys()):
			color_map[node_no] = cvtRGBAflt2HTML(c_map([k100[serial_no]]))
		node_color = color_map.values()
		size_array = size.values()
		node_collection = nx.draw_networkx_nodes(G,pos=pos,node_color=node_color,node_size=size_array,ax=ax,pick_func=pick_func,lamb=lamb)
	elif node_type == "kmeans3_j":
		k3_dict = nx.get_node_attributes(G,"kmeans_3")
		k3 = np.array(k3_dict.values())
		k3 = k3.astype("float32")
		k3 = (k3-k3.min())/(k3.max()-k3.min())

		c_map = cm.jet_r
		color_map = {}
		for serial_no,node_no in enumerate(G.node.keys()):
			color_map[node_no] = cvtRGBAflt2HTML(c_map([k3[serial_no]]))
		node_color = color_map.values()
		size_array = size.values()
		node_collection = nx.draw_networkx_nodes(G,pos=pos,node_color=node_color,node_size=size_array,ax=ax,pick_func=pick_func,lamb=lamb)
	elif node_type == "kmeans3":#クラスタリング結果で可視化(用意したカラーリスト)
		color_map = nx.get_node_attributes(G,"color_k3")
		size_array = size.values()
		node_collection = nx.draw_networkx_nodes(G,pos=pos,node_color=color_map.values(),node_size=size_array,ax=ax,pick_func=pick_func);
	elif node_type == "kmeans100":#クラスタリング結果で可視化(用意したカラーリスト)
		color_map = nx.get_node_attributes(G,"color_k100")
		size_array = size.values()
		node_collection = nx.draw_networkx_nodes(G,pos=pos,node_color=color_map.values(),node_size=size_array,ax=ax,pick_func=pick_func);

	nx.draw_networkx_edges(G,pos,ax=ax)
	return node_collection,color_map

pos = 0
def main(_params):
	global draw_kwargs
	global params
	global pos
	params = _params#本当はこのスクリプト全体をクラスにしてparamsをクラス内変数にしたいが，面倒なのでglobalを使って疑似的にモジュール化

	"""パラメータの読み込み"""
	root_dir = params.get("root_dir")
	exp_name = params.get("exp_name_new")
	nx_dir = params.get("nx_dir")#doc2vecのフォルダ
	src_pkl_name = params.get("src_pkl_name")#doc2vecのファイル（ネットワーク）
	weights_pkl_name = params.get("weights_pkl_name")#doc2vecのファイル（重み）

	"""関連フォルダの存在確認"""
	if not os.path.exists(root_dir):
		print "root_dir",root_dir,"is not exist"
		exit()

	exp_dir = os.path.join(root_dir,exp_name)
	if not os.path.exists(exp_dir):
		print "exp_dir",exp_dir,"is not exist"
		exit()

	# nx_process_dir = os.path.join(nx_dir,"process")
	# if not os.path.exists(nx_process_dir):
	# 	os.mkdir(nx_process_dir)

	"""データの読み込み"""
	with open(os.path.join(nx_dir,src_pkl_name),"r") as fi:
		G = pickle.load(fi)
	with open(os.path.join(nx_dir,weights_pkl_name)) as fi:
		all_nodes_weights = pickle.load(fi)
	with open(os.path.join(exp_dir,"doc2vec.pkl")) as fi:
		d2v = pickle.load(fi)
	print "data_loaded"

	# with open(os.path.join(exp_dir,"instance.pkl")) as fi:
	# 	lda = pickle.load(fi)

	"""パラメータの読み込み"""
	draw_option = params.get("draw_option")

	weight_type = draw_option.get("weight_type",["ATTR","REPUL"])
	weight_attr = draw_option.get("weight_attr",0)
	size_attr = draw_option.get("size_attr",2000)

	pos_rand_path = draw_option.get("pos_rand_path","")
	do_rescale = draw_option.get("do_rescale",True)
	lamb = draw_option.get("lamb",0.5)
	add_random_move = draw_option.get("add_random_move",0.5)

	"""描画パラメータの保存"""
	save_drawoption(draw_option,os.path.join(nx_dir,"draw_option.txt"))

	"""グラフの構築・描画"""
	G_undirected = G#適切なスプリングモデルのためには無向グラフである必要あり
	if G.is_directed():
		G_undirected = G.to_undirected()

	"""bhitsを使うか否かの分岐（使うなら、"to_hosts"と"from_hosts"が必要）"""
	if "BHITS" in weight_type:
		use_bhits = True
		weight_type[weight_type.index("BHITS")] = "HITS"
	else:
		use_bhits = False

	"""引力・斥力計算用のHITSスコア"""
	revised_hits_scores = None
	if ("HITS" in  weight_type) and (type(weight_attr) is dict):
		revised_hits_scores = calc_nodesize(G,attr=weight_attr["type"],min_size=weight_attr["min"],max_size=weight_attr["max"],use_bhits=use_bhits,weight_key="no_weight")#引力斥力計算用に正規化したhitsスコア.calc_nodesizeを共用

	"""初期値の代入"""
	initial_pos = pos_initializer(G_undirected,os.path.join(root_dir,pos_rand_path))
	pos = initial_pos

	"""配置に使うパラメータの設定（引力と斥力）"""
	if "ATTR" not in weight_type:
		all_nodes_weights = None
	weight_label = "weight"
	if "REPUL" not in weight_type :
		weight_label = "no_weight"#各エッジの重みは"weight"キーに入っているため，これ以外を指定すると重みなしとなる．

	"""
	配置の計算
	pos:初期配置
	all_node_weights:全ノード間の重み（斥力計算に用いる）
	weight:エッジの重み（引力計算に用いる）
	rescale:リスケールの有無
	weight_type:重み付けの指定とHITSアルゴリズムを用いるか指定
	revised_hits_scores:引力・斥力計算に用いるHITSスコア
	lamb:引力と斥力の比率．（大きいほど斥力重視）
	add_random_move:配置をランダムに微笑ずらすか否か
	"""
	pos = nx.spring_layout(G_undirected,pos=pos,all_node_weights=all_nodes_weights,weight=weight_label,rescale=do_rescale,weight_type=weight_type,revised_hits_scores=revised_hits_scores,lamb=lamb,add_random_move=add_random_move)#描画位置はここで確定,両方の重みをかける

	"""ノードサイズの計算"""
	if type(size_attr) is dict:
		# size_dict = calc_nodesize(G,attr=size_attr["type"],min_size=size_attr["min"],max_size=size_attr["max"],use_bhits=use_bhits,weight_key="weight")
		size_dict = calc_nodesize(G,attr=size_attr["type"],min_size=size_attr["min"],max_size=size_attr["max"],use_bhits=use_bhits,weight_key="no_weight")
	else:
		size_dict = calc_nodesize(G,attr=size_attr)

	"""実際の描画処理"""
	#new_color_map = draw_network(G,pos,size=size_dict,option=node_type,lda=lda,dpi=dpi,with_label=with_label,lumine=lumine,color_map_by=color_map_by,cmap=cmap)
	draw_kwargs = {
			"size":size_dict,
			"d2v":d2v,
			"draw_option":draw_option
			}

	node_collection,color_map = draw_network(G,pos,**draw_kwargs)
	draw_option["used_color_map"] = color_map

	plot_datas = {"node_collection":node_collection}
	return plot_datas

	"""ネットワークの再保存"""
	# if new_color_map is not None:#カラーマップの更新
	# 	nx.set_node_attributes(G,"color",new_color_map)
	# d=nx.readwrite.json_graph.node_link_data(G)
	# with codecs.open("graph.json","w",encoding="utf8")as fo:
	# 	json.dump(d,fo,indent=4,ensure_ascii=False)
	# with open("final_graph.gpkl","w") as fo:
	# 	pickle.dump(G,fo)

def graph_redraw(G,_pos=None,_color_map=None,**kwargs):
	global pos
	global draw_kwargs
	size = draw_kwargs.get("size")
	draw_option = draw_kwargs.get("draw_option")
	ax = draw_option.get("ax")
	pick_func = draw_option.get("pick_func")
	#widths=kwargs.get("widths")
	if _pos == None:
		_pos = pos
	if _color_map == None:
		_color_map = draw_option.get("used_color_map")
	edges = kwargs.get("edgelist")#if None, nx.draw_network_edges defaulet uses G.edges()

	node_color = _color_map.values()
	size_array = size.values()
	#node_collection=nx.draw(G,pos=_pos,node_color=node_color,node_size=size_array,ax=ax,pick_func=pick_func)
	node_collection = nx.draw_networkx_nodes(G,pos=pos,node_color=node_color,node_size=size_array,ax=ax,pick_func=pick_func)
	nx.draw_networkx_edges(G,pos,ax=ax,edgelist=edges)
	return node_collection

"""
@arg
G:Graph of networkx
node_no:number of the node.collect adjacents around  this node.
link_type:select link direction. in or out.
@ret
set of node numbers
"""
def collect_adjacents(G,node_no,link_type):
	ret_list = []
	edges = []
	if link_type == "in" or link_type == "both":
		for edge in G.in_edges(node_no):
			edges.append(edge)
			ret_list.append(edge[0])
	if link_type == "out" or link_type == "both":
		for edge in G.out_edges(node_no):
			edges.append(edge)
			ret_list.append(edge[1])

	return set(ret_list),edges


"""
change adjacents color to white and hide edges
@arg
G:Graph of networkx
sel_node:number of the node
link_type:select link direction. in or out.
@ret
None
"""
def transparent_adjacents(G,sel_node,link_type,_pos=None,_color_map=None,**kwargs):
	global draw_kwargs
	draw_option = draw_kwargs.get("draw_option")
	if sel_node is None:
		return

	adjacents,edgelist = collect_adjacents(G,sel_node,link_type)
	color_map = draw_option.get("used_color_map")
	new_color_map = {}
	"""該当ノードに対する処理"""
	for k,v in color_map.items():
		if k in adjacents:
			new_color_map[k] = color_map[k]
			#new_color_map[k]=v.replace("#","#00")
		else:
			new_color_map[k] = u"#FFFFFF"
			#new_color_map[k]=v.replace("#","#80")
	new_color_map[sel_node] = color_map[sel_node]

	node_collection = graph_redraw(G,_color_map=new_color_map,edgelist=edgelist)
	return node_collection

"""
draw only nodes within the specified color range
@arg
G:Graph of networkx
lower:minimum value of range
higher:maximum value of range
@ret
None
"""
def cut_off_colors(G,lower,higher,**kwargs):
	global draw_kwargs
	global reg_theta_pca
	draw_option = draw_kwargs.get("draw_option")
	color_map = draw_option.get("used_color_map")

	if higher <= lower:
		lower = 0
		higher = 1

	lda = draw_kwargs.get("lda")
	file_id_dict_inv = {v:k for k, v in lda.file_id_dict.items()}#ファイル名とLDAでの文書番号(逆引き)．LDAの方に作っとけばよかった．．．
	new_color_map = {}
	"""該当ノードに対する処理"""
	edgelist = []
	nodes = color_map.keys()
	for k in nodes:
		val = reg_theta_pca[file_id_dict_inv[k]]
		if lower < val < higher:
			new_color_map[k] = color_map[k]
			edgelist.extend([edge for edge in G.edges(k)])
		else:
			new_color_map[k] = u"#FFFFFF"

	node_collection = graph_redraw(G,_color_map=new_color_map,edgelist=edgelist)
	return node_collection

def recursive_node_crawl(G,tgt_node,link_type,max_cnt,nodes,searched,edges):
	max_cnt -= 1
	if max_cnt < 0:#マイナスになったら終了
		return 0

	adjacents,edgelist=collect_adjacents(G,tgt_node,link_type)
	edgeset = set(edgelist)
	searched.add(tgt_node)
	for node in adjacents:
		if node in searched:
			continue
		recursive_node_crawl(G,node,link_type,max_cnt,nodes,searched,edges)
	nodes.update(adjacents)
	edges.update(edgeset)

"""
対象ノードについて，そのエッジを連鎖的にたどって関わるノードとエッジを全て取得する．
"""
def node_crawler(G,**kwargs):
	global draw_kwargs
	draw_option = draw_kwargs.get("draw_option")
	color_map = draw_option.get("used_color_map")
	link_type = kwargs.get("link_type")
	max_cnt = kwargs.get("max_cnt")
	tgt_node = kwargs.get("tgt_node")
	nodes = set()
	searched = set()
	edges = set()
	recursive_node_crawl(G,tgt_node,link_type,max_cnt=max_cnt,nodes=nodes,searched=searched,edges=edges)
	new_color_map = {}
	"""該当ノードに対する処理"""
	for k,v in color_map.items():
		if k in nodes:
			new_color_map[k] = color_map[k]
		else:
			new_color_map[k] = u"#FFFFFF"
	new_color_map[tgt_node] = color_map[tgt_node]

	node_collection = graph_redraw(G,_color_map=new_color_map,edgelist=edges)
	return node_collection

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
	params = {}
	params["search_word"] = u"Test"
	params["max_page"] = 10
	add_childs = True
	append = False
	save_dir = ur"/home/yukichika/ドキュメント/Data/Search"
	params["root_dir"] = save_dir + suffix_generator_root(params["search_word"],params["max_page"],add_childs,append)

	params["is_largest"] = True
	params["target"] = "myexttext"

	params["K"] = 10
	params["exp_name"] = "K" + unicode(params["K"]) + suffix_generator(params["target"],params["is_largest"])
	params["comp_func_name"] = "comp4_2"

	params["size"] = 100
	params["exp_name_new"] = "D" + unicode(params["size"]) + suffix_generator(params["target"],params["is_largest"])
	params["comp_func_name_new"] = "cos_sim"

	# params["nx_dir"] = os.path.join(os.path.join(params["root_dir"],params["exp_name"]),"nx_datas")
	# params["src_pkl_name"] = "G_with_params_" + params["comp_func_name"] + ".gpkl"
	# params["weights_pkl_name"] = "all_node_weights_" + params["comp_func_name"] + ".gpkl"

	params["nx_dir"] = os.path.join(os.path.join(params["root_dir"],params["exp_name_new"]),"nx_datas")
	params["src_pkl_name"] = "G_with_params_" + params["comp_func_name_new"] + ".gpkl"
	params["weights_pkl_name"] = "all_node_weights_" + params["comp_func_name_new"] + ".gpkl"

	params["draw_option"] = {
		"weight_type":[],

		# "weight_type":["ATTR","REPUL"],

		# "weight_type":["ATTR","REPUL","HITS"],
		# "weight_attr":{"type":"a_score","min":1,"max":3},
		# "size_attr":{"type":"a_score","min":1000,"max":5000},

		# "weight_type":["ATTR","REPUL","HITS"],
		# "weight_attr":{"type":"h_score","min":1,"max":3},
		# "size_attr":{"type":"h_score","min":1000,"max":5000},

		# "weight_type":["ATTR","REPUL","BHITS"],
		# "weight_attr":{"type":"a_score","min":1,"max":3},
		# "size_attr":{"type":"a_score","min":1000,"max":5000},

		# "weight_type":["ATTR","REPUL","BHITS"],
		# "weight_attr":{"type":"h_score","min":1,"max":3},
		# "size_attr":{"type":"h_score","min":1000,"max":5000},

		"lamb":0.5,

		"node_type":"COMP1",
		"cmap":"jet",
		"lumine":200,
		"color_map_by":"vector",

		"pos_rand_path":"nest1.rand",
		"do_rescale":True,
		"with_label":False,
		"add_random_move":False
		}

	main(params)
	# ax = plt.figure().add_subplot(111)
	# draw_option["ax"] = ax
	# params["draw_option"] = draw_option
	# main(params)
	# #ax.get_figure().show()
	# plt.show()

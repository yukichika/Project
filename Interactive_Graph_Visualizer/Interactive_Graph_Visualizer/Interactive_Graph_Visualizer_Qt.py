#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import copy
import cPickle as pickle
import json
import collections
from PyQt4 import QtGui
from PyQt4 import Qt
from PyQt4 import QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#import matplotlib.pyplot as plt
#import numpy as np

import my_graph_drawer

#sys.path.append("../networkx-master")
#import networkx as nx

#ドラッグ初期位置格納用変数の宣言
def zoom_factory(ax,base_scale = 2.,startX=0,starty=0):
	global plot_datas
	def zoom_fun(event):
		# get the current x and y limits
		cur_xlim = ax.get_xlim()
		cur_ylim = ax.get_ylim()
		cur_xrange = (cur_xlim[1] - cur_xlim[0])
		cur_yrange = (cur_ylim[1] - cur_ylim[0])
		xdata = event.xdata # get event x location
		ydata = event.ydata # get event y location
		sizes=plot_datas["node_collection"].get_sizes()
		if event.button == 'up':
			# deal with zoom in
			scale_factor = 1/base_scale
			plot_datas["node_collection"].set_sizes(sizes*base_scale)
		elif event.button == 'down':
			# deal with zoom out
			scale_factor = base_scale
			plot_datas["node_collection"].set_sizes(sizes/base_scale)
		else:
			# deal with something that should never happen
			scale_factor = 1
			print event.button
		# set new limits
		cur_x_rate=(xdata-cur_xlim[0])/cur_xrange
		cur_y_rate=(ydata-cur_ylim[0])/cur_yrange
		new_x_range=[xdata - cur_x_rate*(cur_xrange*scale_factor), xdata + (1-cur_x_rate)*(cur_xrange*scale_factor)]
		new_y_range=[ydata - cur_y_rate*(cur_yrange*scale_factor), ydata + (1-cur_y_rate)*(cur_yrange*scale_factor)]
		ax.set_xlim(new_x_range)
		ax.set_ylim(new_y_range)

		ax.get_figure().canvas.draw()

	def button_press_fun(event):
		global startx
		global starty
		#左クリックの場合以外
		if event.button != 1:
			return

		#ドラッグ開始の初期位置を取得
		startx=event.xdata
		starty=event.ydata

	def drag_fun(event):
		global startx
		global starty
		#左クリックしたまま移動した場合以外
		if event.button != 1:
			return

		# get the current x and y limits
		cur_xlim = ax.get_xlim()
		cur_ylim = ax.get_ylim()
		xdata = event.xdata # get event x location
		ydata = event.ydata # get event y location

		# set new limits
		x_diff=xdata-startx
		y_diff=ydata-starty
		new_x_range=[cur_xlim[0]-x_diff,cur_xlim[1]-x_diff]
		new_y_range=[cur_ylim[0]-y_diff,cur_ylim[1]-y_diff]
		ax.set_xlim(new_x_range)
		ax.set_ylim(new_y_range)
		ax.get_figure().canvas.draw()
		#plt.draw() # force re-draw

	fig = ax.get_figure() # get the figure of interest
	# attach the call back
	fig.canvas.mpl_connect('scroll_event',zoom_fun)
	fig.canvas.mpl_connect('motion_notify_event',drag_fun)
	fig.canvas.mpl_connect('button_press_event',button_press_fun)

	#return the function
	return zoom_fun

flag_dict={}#stores each keys state which pushed or not
"""
キー入力コールバック
b:ノードサイズを倍に
B:ノードサイズを半分に
i:初期状態に戻す
s:現在の描画を保存
"""
def key_press_factory(ax,qwidget):
	global plot_datas
	global initial_plot_datas
	global global_datas
	def key_press_func(event):
		fig = ax.get_figure() # get the figure of interest
		G=global_datas.get("G")
		sizes=plot_datas["node_collection"].get_sizes()
		if event.key == 'b':
			plot_datas["node_collection"].set_sizes(sizes*2)
		if event.key == 'B':
			plot_datas["node_collection"].set_sizes(sizes/2)
		if event.key == 'i':#描画初期化
			initial_sizes=initial_plot_datas["node_sizes"]
			plot_datas["node_collection"].set_sizes(initial_sizes)
		if event.key == 's':
			fig.savefig("graph_figure.png")
		if event.key == 'v':#switch adjacents transparency
			if flag_dict.get("v",True):
				sel_node=global_datas.get("cur_select_node")
				stable_scale_draw(ax,my_graph_drawer.transparent_adjacents,G,sel_node,link_type="both")
				flag_dict["v"]=False
			else:
				stable_scale_draw(ax,my_graph_drawer.graph_redraw,G)
				flag_dict["v"]=True
		if event.key == 'r':#recursive node view
			if flag_dict.get("r",True):
				sel_node=global_datas.get("cur_select_node")
				stable_scale_draw(ax,my_graph_drawer.node_crawler,G,tgt_node=sel_node,max_cnt=3,link_type="in")
				flag_dict["r"]=False
			else:
				stable_scale_draw(ax,my_graph_drawer.graph_redraw,G)
				flag_dict["r"]=True

		#if event.key == 'D':#redraw all nodes
		#	ax.clear()
		#	stable_scale_draw(ax,my_graph_drawer.graph_redraw,G)
		fig.canvas.draw()

	fig = ax.get_figure() # get the figure of interest
	fig.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )#key_press_eventをセットする前にこの2行が必要
	fig.canvas.setFocus()
	fig.canvas.mpl_connect('key_press_event',key_press_func)
	return key_press_func

def denote_words_freq(text,topn=20,delimiter=","):
	count_dict=collections.Counter(text)
	commons=count_dict.most_common(topn)
	ret_str=u""
	for word,cnt in commons:
		ret_str+=unicode(word)+":"+unicode(cnt)+delimiter
	return ret_str[:-1]#最後のdelimiterを除外

initial_plot_datas={}
plot_datas={}
global_datas={}
#cur_select_node=None #stores recently picked node
class ForceGraph():
	def __init__(self,*args,**kwargs):
		parent=kwargs.get("parent")
		self.params=kwargs.get("params")
		self.vervoseWidget=kwargs.get("verboseWidget")

		# Create the mpl Figure and FigCanvas objects.
		self.dpi = 20
		self.fig = Figure((5,4), dpi=self.dpi)
		self.canvas = FigureCanvas(self.fig)    #pass a figure to the canvas
		#self.canvas.setParent(parent)

		"""plt画面に関する設定"""
		self.axes = self.fig.add_subplot(111)
		zoom_factory(self.axes,base_scale=2.)
		key_press_factory(self.axes,args[0])
		self.fig.set_facecolor('w')
		self.axes.axis('equal')#両軸を同じスケールに

		self.params["draw_option"]["ax"]=self.axes
		self.params["draw_option"]["pick_func"]=self.pick_func

		nx_dir=self.params.get("nx_dir")
		src_pkl_name=self.params.get("src_pkl_name")
		with open(os.path.join(nx_dir,src_pkl_name),"r") as fi:
			self.G=pickle.load(fi)
		global global_datas
		global_datas["G"]=self.G

	def on_draw(self):
		global plot_datas
		global initial_plot_datas
		self.axes.clear()
		plot_datas=my_graph_drawer.main(self.params)
		initial_plot_datas["node_sizes"]=plot_datas["node_collection"].get_sizes()
		self.canvas.draw()

	def pick_func(self,event):
		global global_datas
		if event.mouseevent.name != "button_press_event":
			return
		idxs=event.ind
		for i in idxs:
			print idxs,"file_no=",self.G.node.keys()[i]
			#self.vervoseWidget.change_content(i)
			self.vervoseWidget.change_content(self.G.node.keys()[i])
			global_datas["cur_select_node"]=self.G.node.keys()[i]
			break

class VerboseWidget(QtGui.QWidget):
	def __init__(self,*args,**kwargs):
		parent=kwargs.get("parent")
		QtGui.QWidget.__init__(self ,parent=parent)

		self.topicGraph=kwargs.get("topicGraph")
		self.params=kwargs.get("params")
		self.table=QtGui.QTableWidget()
		self.table.horizontalHeader().setVisible(False)
		self.table.verticalHeader().setVisible(False)
		self.table.horizontalHeader().setStretchLastSection(True)
		items_vbox = QtGui.QVBoxLayout(self)
		items_vbox.addWidget(self.table,1)
		self.bow_textbox=QtGui.QTextEdit(self)
		items_vbox.addWidget(self.bow_textbox,0)

		exp_dir=os.path.join(self.params["root_dir"],self.params["exp_name"])
		src_pkl_name=self.params.get("src_pkl_name")
		nx_dir=self.params.get("nx_dir")
		with open(os.path.join(exp_dir,"instance.pkl")) as fi:
		   self.lda=pickle.load(fi)
		with open(os.path.join(nx_dir,src_pkl_name),"r") as fi:
			self.G=pickle.load(fi)

	def change_content(self,file_no):
		src_pages_dir=os.path.join(self.params["root_dir"],"Pages")
		with open(os.path.join(src_pages_dir,unicode(file_no)+".json"),"r") as fj:
			page_info=json.load(fj)

		"""トピックグラフ表示"""
		file_id_dict_inv = {v:k for k, v in self.lda.file_id_dict.items()}#ファイル名とLDAでの文書番号(逆引き)．
		lda_no=file_id_dict_inv.get(file_no)
		if lda_no != None:
			self.topicGraph.on_draw(self.lda.theta()[lda_no])#トピック分布グラフの表示

		tgt_params=[
			["id","LDA_no"],
			["name_id","file_no"],
			["title",u"タイトル"],
			#"len(text)",
			["url",u"url"],
			["domain",u"ドメイン"],
			["len_parents",u"リンクされている数"],
			["len_childs",u"リンクしている数"],
			#"repTopic",
			["auth_score",u"オーソリティスコア"],
			["hub_score",u"ハブスコア"]
			]
		self.table.setRowCount(len(tgt_params))
		self.table.setColumnCount(2)

		for i,(tgt_param,name) in enumerate(tgt_params):
			val=0
			if tgt_param=="id":
				val=lda_no
			elif tgt_param=="name_id":
				val=file_no
			elif tgt_param=="domain":
				url=page_info.get("url")
				val=url.split("/")[2]
			elif tgt_param=="len(text)":
				if(page_info.get("text") != None):
					val=len(page_info.get("text"))
			elif tgt_param=="repTopic":
				val=int(self.lda.n_m_z[id].argmax()+1)
			elif tgt_param=="len_parents":
				val=self.G.in_degree(file_no)#それぞれにアクセスしたいときはG.in_edges(file_no)
			elif tgt_param=="len_childs":
				val=self.G.out_degree(file_no)#それぞれにアクセスしたいときはG.out_edges(file_no)
			elif tgt_param=="auth_score":
				val=self.G.node.get(file_no).get("a_score")
			elif tgt_param=="hub_score":
				val=self.G.node.get(file_no).get("h_score")
			else:
				val=page_info.get(tgt_param)
			if type(val) is not unicode:
				val=unicode(val)
			self.table.setItem(i,0,QtGui.QTableWidgetItem(name))
			self.table.setItem(i,1,QtGui.QTableWidgetItem(val))

		""""表の下に追加する項目"""
		# bow=[unicode(self.lda.vocas[word_id]) for word_id in self.lda.docs[lda_no]]
		bow=[self.lda.vocas[word_id].decode("utf-8") for word_id in self.lda.docs[lda_no]]

		self.bow_textbox.clear()
		self.bow_textbox.append(denote_words_freq(bow,topn=15,delimiter="\n"))

class TopicGraph():
	def __init__(self,*args,**kwargs):
		parent=kwargs.get("parent")
		self.params=kwargs.get("params")

		# Create the mpl Figure and FigCanvas objects.
		self.dpi = 100
		self.fig = Figure((5,4), dpi=self.dpi)
		self.canvas = FigureCanvas(self.fig)    #pass a figure to the canvas
		self.canvas.setParent(parent)

		"""plt画面に関する設定"""
		self.axes = self.fig.add_subplot(111)
		self.fig.set_facecolor('w')

		self.params["draw_option"]["ax"]=self.axes

	def on_draw(self,theta):
		self.axes.clear()
		K=len(theta)
		label_strs=["Topic"+unicode(k+1) for k in range(K+1)]
		self.axes.bar(range(1,K+1),theta,align="center",alpha=0.7)
		self.axes.axis([0,K+1,0,1])
		self.canvas.draw()

def stable_scale_draw(ax,draw_func,*args,**kwargs):
	global plot_datas
	sizes=plot_datas["node_collection"].get_sizes()
	cur_xlim = ax.get_xlim()
	cur_ylim = ax.get_ylim()
	ax.clear()
	node_collection= draw_func(*args,**kwargs)
	ax.get_figure().canvas.draw()
	ax.set_xlim(cur_xlim)
	ax.set_ylim(cur_ylim)
	plot_datas["node_collection"]=node_collection
	plot_datas["node_collection"].set_sizes(sizes)

class SettingWidget(QtGui.QWidget):
	def __init__(self,*setting_args,**kwargs):
		parent=kwargs.get("parent")
		QtGui.QWidget.__init__(self ,parent=parent)

		self.params=kwargs.get("params")

		hbox = QtGui.QHBoxLayout()
		self.attr_weight_check=QtGui.QCheckBox("attr",self)
		self.repl_weight_check=QtGui.QCheckBox("repl",self)
		hbox.addWidget(self.attr_weight_check)    #add canvs to the layout
		hbox.addWidget(self.repl_weight_check)    #add canvs to the layout
		power_group=QtGui.QGroupBox("power")
		power_group.setLayout(hbox)

		vbox = QtGui.QVBoxLayout()

		lower_box=QtGui.QHBoxLayout()
		lower_slider_label = QtGui.QLabel('lower :')
		self.lower_slider=QtGui.QSlider(QtCore.Qt.Horizontal)
		self.lower_slider.setRange(0,100)
		self.lower_slider.setValue(0)
		l_spinbox=QtGui.QSpinBox()
		l_spinbox.setRange(0,100)
		l_spinbox.setValue(0)
		l_spinbox.valueChanged[int].connect(self.lower_slider.setValue)
		self.lower_slider.valueChanged[int].connect(l_spinbox.setValue)
		lower_box.addWidget(lower_slider_label,0)
		lower_box.addWidget(self.lower_slider,1)
		lower_box.addWidget(l_spinbox,0)

		higher_box=QtGui.QHBoxLayout()
		higher_slider_label = QtGui.QLabel('higher:')
		self.higher_slider=QtGui.QSlider(QtCore.Qt.Horizontal)
		self.higher_slider.setRange(0,100)
		self.higher_slider.setValue(100)
		h_spinbox=QtGui.QSpinBox()
		h_spinbox.setRange(0,100)
		h_spinbox.setValue(100)
		h_spinbox.valueChanged[int].connect(self.higher_slider.setValue)
		self.higher_slider.valueChanged[int].connect(h_spinbox.setValue)
		higher_box.addWidget(higher_slider_label,0)
		higher_box.addWidget(self.higher_slider,1)
		higher_box.addWidget(h_spinbox,0)

		vbox.addLayout(lower_box)
		vbox.addLayout(higher_box)
		slider_group=QtGui.QGroupBox("cut off color")
		slider_group.setLayout(vbox)

		apply_button=QtGui.QPushButton("Apply")
		apply_button.clicked.connect(self.apply_button_callback)

		vbox = QtGui.QVBoxLayout(self)
		vbox.addWidget(power_group)
		vbox.addWidget(slider_group)
		vbox.addWidget(apply_button)
		vbox.addStretch(1)

		exp_dir=os.path.join(self.params["root_dir"],self.params["exp_name"])
		src_pkl_name=self.params.get("src_pkl_name")
		nx_dir=self.params.get("nx_dir")
		with open(os.path.join(exp_dir,"instance.pkl")) as fi:
		   self.lda=pickle.load(fi)
		with open(os.path.join(nx_dir,src_pkl_name),"r") as fi:
			self.G=pickle.load(fi)

	def apply_button_callback(self):
		global global_datas
		G=global_datas.get("G")
		ax=self.params["draw_option"]["ax"]
		lower=float(self.lower_slider.value())/self.lower_slider.maximum()
		higher=float(self.higher_slider.value())/self.higher_slider.maximum()

		stable_scale_draw(ax,my_graph_drawer.cut_off_colors,G,lower,higher)

class AppForm(QtGui.QMainWindow):
	def __init__(self,*args,**kwargs):#parent=None,params=None):
		parent=kwargs.get("parent")
		QtGui.QMainWindow.__init__(self, parent)
		self.params=kwargs.get("params")

		self.create_main_window()
		self.forceGraph.on_draw()

	def create_main_window(self):
		self.main_frame = QtGui.QWidget()
		self.topicGraph = TopicGraph(self.main_frame,params=self.params)
		self.verboseWidget=VerboseWidget(self.main_frame,params=self.params,topicGraph=self.topicGraph)
		self.forceGraph = ForceGraph(self.main_frame,params=self.params,verboseWidget=self.verboseWidget)
		self.settingWidget=SettingWidget(self.main_frame,params=self.params)

		#set layout
		hbox = QtGui.QHBoxLayout()
		hbox.addWidget(self.forceGraph.canvas)    #add canvs to the layout

		self.main_frame.setLayout(hbox)

		right_dock1 = QtGui.QDockWidget('Information',self)
		right_dock1.setWidget(self.verboseWidget)
		self.addDockWidget(Qt.Qt.RightDockWidgetArea, right_dock1)

		right_dock2 = QtGui.QDockWidget('Topics',self)
		right_dock2.setWidget(self.topicGraph.canvas)
		self.addDockWidget(Qt.Qt.RightDockWidgetArea, right_dock2)

		left_dock = QtGui.QDockWidget('Settings',self)
		left_dock.setWidget(self.settingWidget)
		self.addDockWidget(Qt.Qt.LeftDockWidgetArea, left_dock)

		self.setCentralWidget(self.main_frame)

		#set widget
		#self.setCentralWidget(self.main_frame)

"""保存名の決定"""
def suffix_generator(target=None,is_largest=False):
	suffix = ""
	if target != None:
		suffix += "_"+target
	if is_largest == True:
		suffix += "_largest"
	return suffix

def main(args):
	params = {}
	params["search_word"] = u"iPhone"
	params["max_page"] = 400
	params["root_dir"] = ur"/home/yukichika/ドキュメント/Data/Search_" + params["search_word"] + "_" + unicode(params["max_page"]) + "_add_childs"

	params["is_largest"] = True
	params["target"] = "myexttext"
	params["K"] = 10
	params["exp_name"] = "K" + unicode(params["K"]) + suffix_generator(params["target"],params["is_largest"])

	params["comp_func_name"] = "comp4_2"
	params["nx_dir"] = os.path.join(os.path.join(params["root_dir"],params["exp_name"]),"nx_datas")
	params["src_pkl_name"] = "G_with_params_" + params["comp_func_name"] + ".gpkl"
	params["weights_pkl_name"] = "all_node_weights_" + params["comp_func_name"] + ".gpkl"



	"""
	weight_type:["ATTR"(斥力計算に重みall_node_weightsを使う),
	　　　　　　　　"REPUL（引力計算にエッジの重みweightを使う）",
	　　　　　　　　"HITS"（HITSアルゴリズムを使う）,
	　　　　　　　　"BHITS"（BHITSアルゴリズムを使う）
				]（オーソリティかハブかはsize_attrとweight_attrで指定）
	weight_attr:引力・斥力計算にHITSを使うか（使うならdictでオーソリティかハブか指定）
	size_attr:ノードの大きさにHITSを使うか（使うならdictでオーソリティかハブか指定）
	pos_rand_path:初期配置の乱数の格納ファイル．（未指定の場合は毎回乱数発生）
	node_type:ノードの着色方法（"REPR"=>代表トピックで着色
						   "REPR2"=>色相をPCAの1次元で，彩度をそれぞれの最大トピックの値で返す
						   "COMP1"=>色相をPCAの1次元で，彩度をそれぞれのトピック分布の各比率で合成(composition)
						   "PIE"=>円グラフでノード表現
						   ）
	do_rescale:リスケールの有無
	with_label:ラベル付与の有無
	lamb:引力と斥力の比率．（大きいほど斥力重視）
	add_random_move:配置をランダムに微笑ずらすか否か
	cmap:色の対応付け方法(カラーバー) "jet" or "lch"
	lumine:lchを用いる場合の輝度
	color_map_by:主成分分析の対象（"phi"=>単語分布
							   "theta"=>トピック分布
							   "pie"=>
							   "None"=>無色）
	"""
	params["draw_option"] = {
		# "weight_type":[],

		"weight_type":["ATTR","REPUL"],

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

		"pos_rand_path":"nest1.rand",
		"node_type":"COMP1",
		"do_rescale":True,
		"with_label":False,
		"lamb":0.5,
		"add_random_move":False,
		"cmap":"jet",
		"lumine":200,
		"color_map_by":"theta"
		}

	app = QtGui.QApplication(args)
	form = AppForm(params=params)
	form.showMaximized()
	sys.exit(app.exec_())

if __name__ == "__main__":
	main(sys.argv)

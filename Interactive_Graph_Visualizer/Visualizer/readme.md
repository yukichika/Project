# Visualizer

## Requirements

This implementation has been tested with the following versions.

```
python 2.7.15
pyqt 4.11.4
matplotlib 1.5.1
pygraphviz 1.3
opencv 2.4.11
scikit-learn 0.18.1
numpy 1.11.3
scipy 0.19.0
```

## How to Use
Interactive_Graph_Visualizer_Qt.pyを使用．  
PyQt4をベースに実装．（doc2vecの結果を反映）  
パラメータはmain文で指定．  

パラメータ
```
weight_type:引力・斥力計算のパラメータ（オーソリティかハブかはweight_attrとsize_attrで指定）  
["ATTR":斥力計算に重みall_node_weightを用いる，"REPUL":引力計算にエッジの重みweightを用いる，"HITS":HITSアルゴリズムを用いる，"BHITS":BHITSを用いる]  

weight_attr:引力・斥力計算にHITSを使うか否か（使うならdictでオーソリティかハブか指定）  
ex."weight_attr":{"type":"a_score","min":1,"max":3}  

size_attr:ノードの大きさにHITSを使うか否か（使うならdictでオーソリティかハブか指定）  
ex."size_attr":{"type":"a_score","min":1000,"max":5000}  

node_type:ノードの着色方法  
（"COMP1":doc2vecのベクトルを主成分分析で圧縮（color_map_byで分岐），"kmeans3":3次元でのクラスタリング結果で着色（カラーリスト），"kmeans100":100次元でのクラスタリング結果で着色（カラーリスト），"kmeans3_j":3次元でのクラスタリング結果で着色（jetカラーマップ），"kmeans100_j":100次元でのクラスタリング結果で着色（jetカラーマップ），"kmeans100_j_sort":100次元でのクラスタリング結果を主成分分析によってソートして着色（jetカラーマップ））  

cmap:色の対応付け方法("jet" or "lch")  
lumine:lchを用いる場合の輝度  
color_map_by:主成分分析の対象（"vector1":doc2vecのベクトルを1次元に圧縮，"vector3":doc2vecのベクトルを3次元に圧縮，"None":無色）  
pos_rand_path:初期配置の乱数の格納ファイル．（未指定の場合は毎回乱数発生）  
do_rescale:リスケールの有無  
with_label:ラベル付与の有無  
lamb:引力と斥力の比率．（大きいほど斥力重視）  
add_random_move:配置をランダムに微笑ずらすか否か  
```

・my_graph_drawer.py  
ノードの配置や着色を行う．  

基本操作  
```
マウスドラッグ：移動  
ホイール操作：ズーム  
クリック：ノードの選択  
b:ノードサイズを大きく  
B:ノードサイズを小さく  
v:隣接ノードのみを表示(もう一度押すと解除)  
r:選択ノードにリンクを張っているノードを3段階まで表示(もう一度押すと解除)  
s:現在の画像を保存(めちゃくちゃサイズでかいので注意)  
i:描画を初期化  
```



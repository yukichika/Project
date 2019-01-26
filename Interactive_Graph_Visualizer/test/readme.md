# test

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
発表時に見せるために，表示させる項目などを変更したもの．  
グラフ構造の表示はInteractive_Graph_Visualizer/Visualizerと同じ．

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

・my_graph_drawer.py  
ノードの配置や着色を行う．  



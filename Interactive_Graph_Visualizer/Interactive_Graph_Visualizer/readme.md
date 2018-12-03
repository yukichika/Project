# Interactive_Graph_Visualizer

## Requirements

This implementation has been tested with the following versions.

```
python 2.7.15
pyqt 4.11.4
matplotlib 1.5.1
opencv 2.4.11
scikit-learn 0.18.1
numpy 1.11.3
scipy 0.19.0
```

## How to Use


--------------------------------------------------------------------------------------
Interactive_Graph_Visualizer_Qt.py
最終的に使用したスクリプト．PyQt4をベースに実装．
main文内のparamsでパラメータを設定.

基本操作
マウスドラッグ：移動
ホイール操作：ズーム
クリック：ノードの選択
b:ノードサイズを大きく
B:ノードサイズを小さく
v:隣接ノードのみを表示(もう一度押すと解除)
r:選択ノードにリンクを張っているノードを3段階まで表示(もう一度押すと解除)
s:現在の画像を保存(めちゃくちゃサイズでかいので注意)
i:描画を初期化

my_graph_drawer.py
Webspace_Visualizerのmake_network_by_nx.pyを改良したもの．
ノードの配置や着色を行う．

Interactive_Graph_Visualizer_d3.py
d3.jsを使って可視化．試作品.

その他のスクリプトについてはWebspace_Visualizerを参照のこと．


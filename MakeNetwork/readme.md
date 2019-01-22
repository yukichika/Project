# MakeNetwork

## Requirements

This implementation has been tested with the following versions.

```
python 2.7.15
matplotlib 1.5.1
mecab-python 0.996
numpy 1.11.3
scipy 0.19.0
```

```
python 3.6.2
gensim 3.2.0
mecab-python 0.996
zenhan 0.5.2
```

## How to Use(for LDA)
収集したWebページに対する可視化前のスクリプト．  
基本的にseries_act.py(python2)を使用．  
テキストの特徴量として，LDAのトピック分布を利用．  

・cvt_to_nxtype2.py  
収集したWebページのデータ（json形式）をnetworkxの形式に変換．  

・LDA_for_SS.py  
名詞の抽出(jsonからchasen)とLDAの実行（chasenからLDA）．  

・LDA_modify_for_graph.py  
LDAから，ノードの代表トピック・色とエッジ間の重みを取得しnetworkxに反映．  
また，全ノード間の重み（距離）も計算して保存．

・arrange_G_data.py  
bhitsを計算するために，リンク先・リンク元のドメインをGに反映.  

・calc_HITS.py  
ノードのHITSスコアを計算して，neworkxに反映．  


## How to Use(for Doc2vec)
LDAでの結果をDoc2vecの結果に置き換えるスクリプト．  
D2V_for_SS.py(python3)→D2V_modify_for_graph.py(python2)→cluster.py(python2)の順に回す．  
テキストの特徴量として，Doc2vecのベクトルを利用．  

・Preprocessing.py(python3)  
Doc2vecでベクトル化するための前処理プログラム．  

・D2V_for_SS.py(python3)  
LDAで解析したWebページのみ，前処理を行い，学習済みモデルを用いてベクトル化．  
Webページのidがキー，numpy型のベクトルが要素の辞書で保存．  

・check_word.py(python3)  
webページ集合の語彙数・平均単語数をテキストファイルに保存．  
D2V_for_SS.pyのときにやっとけばよかった...  

・D2V_modify_for_graph.py(python2)  
エッジ間の重みをDoc2vecのベクトルで取得しnetworkxに反映．  
また，全ノード間の重み（距離）も計算して保存．  
（コサイン類似度とユークリッド距離の2パターン）

・cluster.py(python2)  
doc2vecのベクトルでクラスタリングして，クラスタの情報（クラスタ番号と色）をグラフに反映．  
グラフの次元数を保持したままクラスタリングした場合と，主成分分析を用いて次元圧縮してからクラスタリングした場合どちらも反映．  
各クラスタの重心もpklファイルに保存．  

・check_distance.py(python2)  
Webページ間の類似度を算出．  
```
・コサイン類似度（0～1に正規化）
・ユークリッド距離（0～1に正規化）  
・指数で正規化したユークリッド距離  
```

・check_distance_distribution.py(python2)  
可視化するWebページ間の類似度の分布（自分自身は除く）を保存．  
```
・コサイン類似度  
・コサイン類似度(0～1に正規化)  
・ユークリッド距離  
・ユークリッド距離(0~1に正規化)  
・ユークリッド距離(0~1に正規化し，反転)  
・指数で正規化したユークリッド距離  
```


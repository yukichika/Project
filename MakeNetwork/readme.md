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
収集したWebページに対する可視化前のスクリプト．(python2)  
基本的にseries_act.pyを使用．  
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
LDAでの結果をDoc2vecの結果に置き換えるスクリプト．(python3)  
D2V_for_SS.py→D2V_modify_for_graph.pyの順に回す．  
テキストの特徴量として，Doc2vecのベクトルを利用．  

・Preprocessing.py(python3)  
Doc2vecでベクトル化するための前処理プログラム．  

・D2V_for_SS.py(python3)  
LDAで解析したWebページのみ，前処理を行い，学習済みモデルを用いてベクトル化．  
Webページのidがキー，numpy型のベクトルが要素の辞書で保存．  

・D2V_modify_for_graph.py(python2)  
エッジ間の重みをDoc2vecのベクトルで取得しnetworkxに反映．  
また，全ノード間の重み（距離）も計算して保存．  

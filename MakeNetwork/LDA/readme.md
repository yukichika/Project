# MakeNetwork/LDA

## Requirements

This implementation has been tested with the following versions.

```
python 2.7.15
matplotlib 1.5.1
mecab-python 0.996
numpy 1.11.3
scipy 0.19.0
```

## How to Use

収集したWebページに対する可視化前のスクリプト．  
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



# Webspace_Visualizer

## Requirements

This implementation has been tested with the following versions.

```
python 2.7.15
matplotlib 1.5.1
mecab-python 0.996
numpy 1.11.3
scipy 0.19.0
tqdm 4.28.1
```

## How to Use

収集したWebページに対する可視化前のスクリプト．  
基本的にseries_act.pyを使用．  

cvt_to_nxtype2.py  
収集したWebおエージのデータ（json形式）をnetworkxの形式に変換．  

LDA_for_SS.py  
LDAの実行．  

LDA_modify_for_graph.py  
LDAから，ノードの代表トピック・色とエッジ間の重みを取得しnetworkxに反映．  
また，全ノード間の重み（距離）も計算して保存．


arrange_G_data.py  
bhitsを計算するために，リンク先・リンク元のドメインをGに反映.  

calc_HITS.py  
ノードのHITSスコアを計算して，neworkxに反映．  

実行手順としてはseries_act.py→arrange_G_data.py→calc_HITS.py


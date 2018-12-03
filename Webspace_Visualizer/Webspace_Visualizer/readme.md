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


--------------------------------------------------------------------------------------

収集したWebページの可視化スクリプト
基本的にseries_actを使用．
パラメータはmain関数内で設定．
各スクリプト実行時に生成データを格納するフォルダを作成し
既にフォルダが存在した場合はスキップして次の処理に進む．
なお途中で異常終了した場合，生成したフォルダを削除する処理がないため，その場合は手動で削除すること．

cvt_to_nxtype2.py
収集したWebページのデータ(各json形式)をnetworkXの形式に変換する．

LDA_for_SS.py
可視化用のLDAの実行.

LDA_modify_for_graph.py
LDAの結果から類似度等を計算し，networkxに反映させられる形に変形．

make_network_by_nx.py
make_network_by_nx.main()で収集したデータの可視化

series_act.py
上記処理を順番に実行する．基本的にはこれ一つで済むはず．

collection_annalizer2.py
収集結果の解析用データを出力．xlsx形式．
収集時のidやファイル名，タイトル，url，ドメイン，トピック分布，各トピックの内訳などを出力．

LDA_PCA.py
LDA結果に対するPCA処理をまとめたもの

color_changer.py
lab,lchなどの色変換を行う．なおhtml形式の色表現への変換はmake_network_by_nx.pyにある．

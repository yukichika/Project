# MakeNetwork_D2V

## Requirements

This implementation has been tested with the following versions.

```
python 3.6.2
gensim 3.2.0
mecab-python 0.996
zenhan 0.5.2
```

```
python 2.7.15
numpy 1.11.3
matplotlib 1.5.1
```

## How to Use
Preprocess_for_D2V.py→D2V_modify_for_graph.pyの順に回す．  

Preprocess_for_D2V.py(python3)  
MakeNetwork_LDAで最終的に使用したWebページの前処理とベクトル化．  

Preprocessing.py(python3)  
Preprocess_for_D2V.pyで用いる文書に対する前処理の関数をまとめたプログラム．  

D2V_modify_for_graph.py(python2)  
MakeNetwork_LDAで最終的に使用したネットワークGのエッジ間の重みをDoc2vecベクトル同士のコサイン類似度に置き換え．  
また，全ノード間の重み（距離）も計算して保存．  


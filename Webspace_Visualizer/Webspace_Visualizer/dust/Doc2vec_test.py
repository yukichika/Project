#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim import models

INPUT_MODEL = u"/home/yukichika/ドキュメント/Doc2vec_model/Wikipedia1012155_dm_20_w5_m5_20.model"
model = models.Doc2Vec.load(INPUT_MODEL)

print("文書数：" + str(model.corpus_count))
print("語彙数：" + str(len(model.wv.vocab)))

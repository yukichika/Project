#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os

"""
python2：str型=>unicode型
python3：bytes型=>str型
"""
def tounicode(data):
    f = lambda d, enc: d.decode(enc)
    codecs = ['shift_jis','utf-8','euc_jp','cp932',
              'euc_jis_2004','euc_jisx0213','iso2022_jp','iso2022_jp_1',
              'iso2022_jp_2','iso2022_jp_2004','iso2022_jp_3','iso2022_jp_ext',
              'shift_jis_2004','shift_jisx0213','utf_16','utf_16_be',
              'utf_16_le','utf_7','utf_8_sig']

    for codec in codecs:
        try: return f(data, codec)
        except: continue
    return None

"""
Sort the given list in the way that humans expect.
"""
def sort_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

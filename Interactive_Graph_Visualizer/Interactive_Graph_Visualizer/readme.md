# Interactive_Graph_Visualizer

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
Interactive_Graph_Visualizer_Qt.py���g�p�D  
PyQt4���x�[�X�Ɏ����D�iLDA�̌��ʂ𔽉f�������z�u�j  
�p�����[�^��main���Ŏw��D  

�p�����[�^
```
weight_type:���́E�˗͌v�Z�̃p�����[�^�i�I�[�\���e�B���n�u����weight_attr��size_attr�Ŏw��j  
["ATTR":�˗͌v�Z�ɏd��all_node_weight��p����C"REPUL":���͌v�Z�ɃG�b�W�̏d��weight��p����C"HITS":HITS�A���S���Y����p����C"BHITS":BHITS��p����]  

weight_attr:���́E�˗͌v�Z��HITS���g�����ۂ��i�g���Ȃ�dict�ŃI�[�\���e�B���n�u���w��j  
ex."weight_attr":{"type":"a_score","min":1,"max":3}  

size_attr:�m�[�h�̑傫����HITS���g�����ۂ��i�g���Ȃ�dict�ŃI�[�\���e�B���n�u���w��j  
ex."size_attr":{"type":"a_score","min":1000,"max":5000}  

cmap:�F�̑Ή��t�����@("jet" or "lch")  
lumine:lch��p����ꍇ�̋P�x  
color_map_by:�听�����͂̑Ώہi"phi":�P�ꕪ�z�C"theta":�g�s�b�N���z�C"None":���F�j  
pos_rand_path:�����z�u�̗����̊i�[�t�@�C���D�i���w��̏ꍇ�͖��񗐐������j  
do_rescale:���X�P�[���̗L��  
with_label:���x���t�^�̗L��  
lamb:���͂Ɛ˗͂̔䗦�D�i�傫���قǐ˗͏d���j  
add_random_move:�z�u�������_���ɔ��΂��炷���ۂ�  
```

�Emy_graph_drawer.py  
�m�[�h�̔z�u�⒅�F���s���D  

��{����  
```
�}�E�X�h���b�O�F�ړ�  
�z�C�[������F�Y�[��  
�N���b�N�F�m�[�h�̑I��  
b:�m�[�h�T�C�Y��傫��  
B:�m�[�h�T�C�Y��������  
v:�אڃm�[�h�݂̂�\��(������x�����Ɖ���)  
r:�I���m�[�h�Ƀ����N�𒣂��Ă���m�[�h��3�i�K�܂ŕ\��(������x�����Ɖ���)  
s:���݂̉摜��ۑ�(�߂��Ⴍ����T�C�Y�ł����̂Œ���)  
i:�`���������  
```


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
���W����Web�y�[�W�ɑ΂�������O�̃X�N���v�g�D  
��{�I��series_act.py(python2)���g�p�D  
�e�L�X�g�̓����ʂƂ��āCLDA�̃g�s�b�N���z�𗘗p�D  

�Ecvt_to_nxtype2.py  
���W����Web�y�[�W�̃f�[�^�ijson�`���j��networkx�̌`���ɕϊ��D  

�ELDA_for_SS.py  
�����̒��o(json����chasen)��LDA�̎��s�ichasen����LDA�j�D  

�ELDA_modify_for_graph.py  
LDA����C�m�[�h�̑�\�g�s�b�N�E�F�ƃG�b�W�Ԃ̏d�݂��擾��networkx�ɔ��f�D  
�܂��C�S�m�[�h�Ԃ̏d�݁i�����j���v�Z���ĕۑ��D

�Earrange_G_data.py  
bhits���v�Z���邽�߂ɁC�����N��E�����N���̃h���C����G�ɔ��f.  

�Ecalc_HITS.py  
�m�[�h��HITS�X�R�A���v�Z���āCneworkx�ɔ��f�D  


## How to Use(for Doc2vec)
LDA�ł̌��ʂ�Doc2vec�̌��ʂɒu��������X�N���v�g�D  
D2V_for_SS.py(python3)��D2V_modify_for_graph.py(python2)��cluster.py(python2)�̏��ɉ񂷁D  
�e�L�X�g�̓����ʂƂ��āCDoc2vec�̃x�N�g���𗘗p�D  

�EPreprocessing.py(python3)  
Doc2vec�Ńx�N�g�������邽�߂̑O�����v���O�����D  

�ED2V_for_SS.py(python3)  
LDA�ŉ�͂���Web�y�[�W�̂݁C�O�������s���C�w�K�ς݃��f����p���ăx�N�g�����D  
Web�y�[�W��id���L�[�Cnumpy�^�̃x�N�g�����v�f�̎����ŕۑ��D  

�ED2V_modify_for_graph.py(python2)  
�G�b�W�Ԃ̏d�݂�Doc2vec�̃x�N�g���Ŏ擾��networkx�ɔ��f�D  
�܂��C�S�m�[�h�Ԃ̏d�݁i�����j���v�Z���ĕۑ��D  
�i�R�T�C���ގ��x�ƃ��[�N���b�h������2�p�^�[���j

�Ecluster.py(python2)  
doc2vec�̃x�N�g���ŃN���X�^�����O���āC�N���X�^�̏����O���t�ɔ��f�D  
�O���t�̎�������ێ������܂܃N���X�^�����O�����ꍇ�ƁC�听�����͂�p���Ď������k���Ă���N���X�^�����O�����ꍇ�ǂ�������f�D  

�Echeck_distance.py(python2)  
Web�y�[�W�Ԃ̗ގ��x���R�T�C���ގ��x�i���K���j�E���[�N���b�h�����i���K���j�ŎZ�o�D  

�Echeck_distance_distribution.py(python2)  
��������Web�y�[�W�Ԃ̗ގ��x�̕��z�i�������g�͏����j��ۑ��D  
�R�T�C���ގ��x�E�R�T�C���ގ��x(���K��)�E���[�N���b�h�����E���[�N���b�h����(���K��)�E�w���Ő��K���������[�N���b�h�������g�p�D  

�Ewikipedia_distance.py(python3)  
doc2vec�̃x�N�g���̗ގ��x���R�T�C���ގ��x�E���[�N���b�h�����E�w���Ő��K���������[�N���b�h�����ŎZ�o�D  
����������Web�y�[�W�͎g���Ă��Ȃ����߁C�e�X�g�p�iWikipedia���g�p�j  

�Ewikipedia_cal.py(python3)  
doc2vec�̃x�N�g���̉��Z���R�T�C���ގ��x�E���[�N���b�h�����E�w���Ő��K���������[�N���b�h�����ŎZ�o�D  
����������Web�y�[�W�͎g���Ă��Ȃ����߁C�e�X�g�p�iWikipedia���g�p�j  



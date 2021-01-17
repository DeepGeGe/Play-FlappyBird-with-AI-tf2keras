# -*- coding: utf-8 -*-
# @Time    : 2021/1/10 15:05
# @Author  : He Ruizhi
# @File    : test.py
# @Software: PyCharm

# import numpy as np
#
# x1 = np.arange(16).reshape((4,4))
# x2 = np.arange(10,19,1).reshape((3,3))
#
# y2 = np.stack((x1,x1,x1,x1),axis=2)
#
# print(x1)
# print(y2)


from PIL import Image
import numpy as np
import glob

for faces_path in glob.glob(r'assets/sprites/redbird-upflap.png'):
    a = np.asarray(Image.open(faces_path))
    im = Image.fromarray(a.astype('uint8'))
    im = im.resize((16, 16), Image.ANTIALIAS)
    im.save(r'assets/sprites/ico.png')

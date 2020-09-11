# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : information.py
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/8/17 下午3:19
import matplotlib.pyplot as plt
import numpy as np

x = range(28)
y_cer = [37.54, 27.10, 23.21, 21.11, 20.13, 18.08, 16.99, 16.50, 15.39, 15.25,
         14.89, 13.15, 12.46, 12.38, 12.22, 12.17, 11.94, 11.63, 11.58, 11.52,
         11.50, 11.50, 11.45, 11.55, 11.44, 11.46, 11.42, 11.40]
y_loss = [26.70, 9.96, 7.73, 6.64, 5.96, 5.50, 5.08, 4.78, 4.53, 4.32,
          4.15, 3.60, 3.24, 3.14, 3.07, 3.01, 2.86, 2.74, 2.66, 2.61,
          2.58, 2.56, 2.56, 2.61, 2.58, 2.55, 2.53, 2.52]
# plt.xticks(range(0, 28, 1))
# plt.yticks(range(0, 30, 1))
# plt.title('epoch-loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.plot(x, y_loss)

plt.xticks(range(0, 28, 1))
plt.yticks(range(0, 40, 1))
plt.title('epoch-cer')
plt.xlabel('epoch')
plt.ylabel('cer')
plt.plot(x, y_cer)
plt.show()

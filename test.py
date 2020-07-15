"""
@File:test.py
@Desciption:
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2020/7/5 下午3:04 
"""
import numpy as np
a = np.empty((0,),dtype="float32")
b = np.random.randn(5,)
c = np.concatenate((a,b),axis=0)
print(a.shape)
print(a)
print(b.shape)
print(b)
print(c.shape)
print(c)
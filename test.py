"""
@File:test.py
@Desciption:
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2020/7/5 下午3:04 
"""
# from audio.streamRec_unlimit_dynamic_window import streaming_asr
#
# streaming_asr()
import heapq
import numpy as np

a = np.zeros((5,), dtype=float)
b = np.random.random(size=(5, 5))
print(b)
print(a.argmax())
print(b.argmax())
print(heapq.nlargest(5, range(25), b.take))

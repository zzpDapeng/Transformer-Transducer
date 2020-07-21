"""
@File:test.py
@Desciption:
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2020/7/5 下午3:04 
"""

# s = "5_3099_20170702141442 播 放 西 野 卡 纳 的 歌\n"
# a = s.strip().split()
# content = a[1:]
# print(a)
# import string
# from zhon.hanzi import punctuation
#
# ct = punctuation
# et = string.punctuation
# token = ct + et
#
# def is_ok(ch):
#     """判断一个unicode是否是汉字"""
#     global token
#     if ch in token:
#         return False
#     elif '\u4e00' <= ch <= '\u9fff':
#         return True
#     return False
#
# print(is_ok("西"))
# s = " a "
# s.strip()
# print(s)

# import editdistance
# def computer_cer(preds, labels):
#     dist = sum(editdistance.eval(label, pred) for label, pred in zip(labels, preds))
#     total = sum(len(l) for l in labels)
#     return dist, total
#
# labels = [['你','好','在','吗'],['我','呵','呵','了']]
# preds = [['你','在'],['我','呵','了','哦']]
# print(computer_cer(preds, labels))

from tt.utils import dict_map,write_result
pred = [[1,2],[2,1]]
trans = [[1,2],[1,2]]
dic = {1:"你",2:"在"}
pred = dict_map(pred,dic)
trans = dict_map(trans,dic)
write_result(pred,trans)



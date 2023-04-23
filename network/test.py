# def get(method):
#     assert method in ['low1','low2','low4','low8','low16','low32']
#     num = int(method[3:])
#     if 'low' in method:
#         print('yes')
#     else:
#         print('no')
#     return num
# print(get(method='low4'))
#
# import torch
# a = torch.zeros(2, 3, 5)
# print(a)
import numpy as np
import torch
# list = [[[115, 125, 130],
#          [40,  68,  142],
#          [165, 30,   75]]]
# print(np.array(list).shape)
# tensor = torch.Tensor(list)
# print(tensor)
#
# transposed_tensor = tensor.view(1, -1)

# "回文数"
# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         sign = 0
#         if x < 0:
#             x = abs(x)
#             sign = -1
#         huiwen = int(str(x)[::-1]) * sign
#         if -2**31 <= x <= 2**31 - 1:
#             if huiwen == x:
#                 return True
#             else:
#                 return False
#         else:
#             return False
# "罗马数字转整数"
# dict = {'M':1000, 'CM':900, 'CD':400, 'D':500, 'XC':90, 'IV':4, 'IX':9, 'V':5, 'I':1, 'C':100, 'L':50, 'XL':40, 'X':10, 'XI':11}
# value = 0
# s = "DCXXI"
# j = len(s)
# for i in dict:
#     if 1 <= len(s) <= 15:
#         if i in s:
#             t = len(i)
#             j = j - t
#             print(j)
#             if j >= 0:
#                 value = value + dict[i]
#                 print(value)
# if j >= 0:
#     value = value + j
# print(value)

import torch.nn as nn
a = torch.rand(1,3,8,8)
print(a)
b = a[:, :, 1::2, 1::2]
print(b)
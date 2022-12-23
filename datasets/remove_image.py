import os
import glob
import json

f = open('/mmlabworkspace/WorkSpaces/danhnt/tuyensh/tungphan/AIC/VN-ABINet/datasets/vintext/train.json','r')
data = json.load(f)
list_img = []
for i in range(len(data['images'])):
    file_name = data['images'][i]['file_name']
    list_img.append(file_name)
print(len(list_img))
#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import sys
import cv2
import numpy as np
from shapely.geometry import *

cV2 = [' ',"'",'^', '\\', '}', 'ỵ','Ỵ', '>', '<', '{', '~', '`', '°', '$', 'ẽ', 'ỷ', 'ẳ', '_', 'ỡ', ';', '=', 'Ẳ', 'j', '[', ']', 'ẵ', '?', 'ẫ', 'Ẵ', 'ỳ', 'Ỡ', 'ẹ', 'è', 'z', 'ỹ', 'ằ', 'õ', 'ũ', 'Ẽ', 'ỗ', 'ỏ', '@', 'Ằ', 'Ỳ', 'Ẫ', 'ù', 'ử', '#', 'Ẹ', 'Z', 'Õ', 'ĩ', 'Ỏ', 'È', 'Ỷ', 'ý', 'Ũ', '*', 'ò', 'é', 'q', 'ở', 'ổ', 'ủ', 'ẩ', 'ã', 'ẻ', 'J', 'ữ', 'ễ', 'ặ', '+', 'ứ', 'Ỹ', 'ự', 'ụ', 'Ỗ', '%', 'ắ', 'ồ', '"', 'ề', 'ể', 'ỉ', 'ợ', '!', 'Ẻ', 'ừ', 'ọ', '&', 'ì', 'É', 'ậ', 'Ù', 'Ặ', 'x', 'Ỉ', 'ú', 'í', 'ó', 'Ẩ', 'ị', 'ế', 'Ứ', 'â', 'ấ', 'ầ', 'ớ', 'ă', 'Ủ', 'Ĩ', '(', 'Ắ', 'Ừ', ')', 'ờ', 'Ý', 'Ễ', 'Ã', 'ô', 'ộ', 'Ữ', 'Ợ', 'ả', 'Ở', 'ệ', 'W', 'ơ', 'Ổ', 'ố', 'Ề', 'f', 'Ử', 'ạ', 'w', 'Ò', 'Ự', 'Ụ', 'Ú', 'Ồ', 'ê', 'Ó', 'Ì', 'b', 'Í', 'Ể', 'đ', 'Ớ', '/', 'k', 'Ă', 'v', 'Ị', 'Ậ', 'Ọ', 'd', 'Ầ', 'Ấ', 'ư', 'á', 'Ế', 'p', 'Ơ', 'F', 'Ả', 'Ộ', 'Ê', 'Ờ', 's', '-', 'à', 'y', 'Ố', 'l', 'Â', 'Q', ',', 'X', 'Ệ', 'Ạ', 'Ô', 'r', ':', '6', '7', 'u', '4', 'm', '5', 'e', '8', 'c', 'Ư', 'Á', '9', 'D', '3', 'o', '.', 'Y', 'g', 'K', 'a', 'À', 't', '2', 'B', 'E', 'V', 'R', '1', 'S', 'i', 'L', 'P', 'Đ', 'h', 'U', '0', 'M', 'O', 'n', 'A', 'G', 'I', 'C', 'T', 'H', 'N']


if len(sys.argv) < 3:
  print("Usage: python convert_to_detectron_json.py root_path phase split")
  print("For example: python convert_to_detectron_json.py data train 100200")
  exit(1)
root_path = sys.argv[1]
phase = sys.argv[2]
split = int(float(sys.argv[3]))
dataset = {
    'licenses': [],
    'info': {},
    'categories': [],
    'images': [],
    'annotations': []
}
with open(os.path.join(root_path, 'classes.txt')) as f:
  classes = f.read().strip().split()
for i, cls in enumerate(classes, 1):
  dataset['categories'].append({
      'id': i,
      'name': cls,
      'supercategory': 'beverage',
      'keypoints': ['mean',
                    'xmin',
                    'x2',
                    'x3',
                    'xmax',
                    'ymin',
                    'y2',
                    'y3',
                    'ymax',
                    'cross']  # only for keypoints
  })

def get_category_id(cls):
  for category in dataset['categories']:
    if category['name'] == cls:
      return category['id']


_indexes = sorted([f.split('.')[0]
for f in os.listdir(os.path.join(root_path, 'newlabel'))])
# _indexes = _indexes[1:]
# print(_indexes)
# exit()
if phase == 'tn':
  indexes = [line for line in _indexes if int(
      line) >= split]  # only for this file
else:
  indexes = [line for line in _indexes if int(line) <= split]
j = 1
print(_indexes)
for index in indexes:
  print('Processing: ' + index)
  im = cv2.imread(os.path.join(root_path, 'ctwtrain_text_image/') + index + '.jpg')
  height, width, _ = im.shape
  dataset['images'].append({
      'coco_url': '',
      'date_captured': '',
      'file_name': index + '.jpg',
      'flickr_url': '',
      'id': int(index),
      'license': 0,
      'width': width,
      'height': height
  })
  anno_file = os.path.join(root_path, 'newlabel/') + index + '.txt'

  with open(anno_file) as f:
    lines = [line for line in f.readlines() if line.strip()]
    for i, line in enumerate(lines):
      pttt = line.strip().split('||||')
      parts = pttt[0].split(',')
      ct = pttt[-1].strip()
  
      cls = 'text'
      segs = [float(kkpart) for kkpart in parts[:16]]  
      
      xt = [segs[ikpart] for ikpart in range(0, len(segs), 2)]
      yt = [segs[ikpart] for ikpart in range(1, len(segs), 2)]
      xmin = min([xt[0],xt[3],xt[4],xt[7]])
      ymin = min([yt[0],yt[3],yt[4],yt[7]])
      xmax = max([xt[0],xt[3],xt[4],xt[7]])
      ymax = max([yt[0],yt[3],yt[4],yt[7]])
      width = max(0, xmax - xmin + 1)
      height = max(0, ymax - ymin + 1)
      if width == 0 or height == 0:
        continue

      max_len = 100
      recs = [len(cV2)+1 for ir in range(max_len)]
      # recs = []
      print(ct)
      # ct =  ct.decode("UTF-8")
      # print('ct', ct)
      
      for ix, ict in enumerate(ct):        
        if ix >= max_len: continue
        if ict in cV2:
            recs[ix] = cV2.index(ict)
        else:
          recs[ix] = len(cV2)

      dataset['annotations'].append({
          'area': width * height,
          'bbox': [xmin, ymin, width, height],
          'category_id': get_category_id(cls),
          'id': j,
          'image_id': int(index),
          'iscrowd': 0,
          'bezier_pts': segs,
          'rec': recs
      })
      j += 1
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
  json.dump(dataset, f)

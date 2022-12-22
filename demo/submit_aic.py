from utils.detector import Detector
from detectron2.data.detection_utils import read_image
from distutils.command.config import config
import torch
import numpy as np
import os
import glob

CTLABELS = [' ',"'",'^', '\\', '}', 'ỵ','Ỵ', '>', '<', '{', '~', '`', '°', '$', 'ẽ', 'ỷ', 'ẳ', '_', 'ỡ', ';', '=', 'Ẳ', 'j', '[', ']', 'ẵ', '?', 'ẫ', 'Ẵ', 'ỳ', 'Ỡ', 'ẹ', 'è', 'z', 'ỹ', 'ằ', 'õ', 'ũ', 'Ẽ', 'ỗ', 'ỏ', '@', 'Ằ', 'Ỳ', 'Ẫ', 'ù', 'ử', '#', 'Ẹ', 'Z', 'Õ', 'ĩ', 'Ỏ', 'È', 'Ỷ', 'ý', 'Ũ', '*', 'ò', 'é', 'q', 'ở', 'ổ', 'ủ', 'ẩ', 'ã', 'ẻ', 'J', 'ữ', 'ễ', 'ặ', '+', 'ứ', 'Ỹ', 'ự', 'ụ', 'Ỗ', '%', 'ắ', 'ồ', '"', 'ề', 'ể', 'ỉ', 'ợ', '!', 'Ẻ', 'ừ', 'ọ', '&', 'ì', 'É', 'ậ', 'Ù', 'Ặ', 'x', 'Ỉ', 'ú', 'í', 'ó', 'Ẩ', 'ị', 'ế', 'Ứ', 'â', 'ấ', 'ầ', 'ớ', 'ă', 'Ủ', 'Ĩ', '(', 'Ắ', 'Ừ', ')', 'ờ', 'Ý', 'Ễ', 'Ã', 'ô', 'ộ', 'Ữ', 'Ợ', 'ả', 'Ở', 'ệ', 'W', 'ơ', 'Ổ', 'ố', 'Ề', 'f', 'Ử', 'ạ', 'w', 'Ò', 'Ự', 'Ụ', 'Ú', 'Ồ', 'ê', 'Ó', 'Ì', 'b', 'Í', 'Ể', 'đ', 'Ớ', '/', 'k', 'Ă', 'v', 'Ị', 'Ậ', 'Ọ', 'd', 'Ầ', 'Ấ', 'ư', 'á', 'Ế', 'p', 'Ơ', 'F', 'Ả', 'Ộ', 'Ê', 'Ờ', 's', '-', 'à', 'y', 'Ố', 'l', 'Â', 'Q', ',', 'X', 'Ệ', 'Ạ', 'Ô', 'r', ':', '6', '7', 'u', '4', 'm', '5', 'e', '8', 'c', 'Ư', 'Á', '9', 'D', '3', 'o', '.', 'Y', 'g', 'K', 'a', 'À', 't', '2', 'B', 'E', 'V', 'R', '1', 'S', 'i', 'L', 'P', 'Đ', 'h', 'U', '0', 'M', 'O', 'n', 'A', 'G', 'I', 'C', 'T', 'H', 'N']
voc_size = 230
def decode(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < voc_size:
            s += CTLABELS[c]
    return s

detector = Detector('./configs/ABINet/VinText.yaml', './model_0059999.pth')
list_img = glob.glob("/mmlabworkspace/WorkSpaces/danhnt/tuyensh/tungphan/AIC/VN-ABINet/uaic2022_public_valid/images/*")
output_dir = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/tungphan/AIC/VN-ABINet/output_aic"
for img_path in list_img:
    name = os.path.basename(img_path)
    name = name[:-4] + ".txt"
    f = open(os.path.join(output_dir,name),"a+")
    img = read_image(img_path, format='BGR')
    prediction, vis = detector.predict(img)
    instances = prediction["instances"]
    scores = instances.scores.tolist()
    if hasattr(instances, 'beziers'):
        beziers = instances.beziers.cpu().numpy()
    else:
        beziers = instances.pred_boxes.tensor.numpy()
    recs = instances.recs.cpu().numpy()
    result = ""
    for bezier, rec, score in zip(beziers, recs, scores):
        if score >= 0.5:
            b = bezier
            poly = str(int(b[0])) + "," + str(int(b[1])) + "," +  str(int(b[6]))  + "," +  str(int(b[7]))  + "," +  str(int(b[8]))  + "," +  str(int(b[9]))  + "," +  str(int(b[14]))  + "," +  str(int(b[15]))
            s = decode(rec)
            result = poly + "," + s +"\n"
            f.write(result)
    f.close()






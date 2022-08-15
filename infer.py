import glob
import numpy as np
import threading
import time
import random
import os
import base64
import cv2
import json
import paddlex as pdx
def infer_1(picture_path):
    picture_path=picture_path
    predictor = pdx.deploy.Predictor('./model1')

    result = predictor.predict(picture_path)
    pdx.det.visualize(picture_path, result, threshold=0.5, save_dir='./picture_infer')

# 读取图片与获取预测结果
    model = pdx.load_model('./model1')
    img = cv2.imread(picture_path)
    result = model.predict(img)

    keep_results = []
    areas = []
    f = open('result.txt', 'a')
    count = 0
    for dt in np.array(result):
        cname, bbox, score = dt['category'], dt['bbox'], dt['score']
        if score < 0.5:
            continue
        keep_results.append(dt)
        count += 1
        f.write(str(dt) + '\n')
        f.write('\n')
        areas.append(bbox[2] * bbox[3])
    areas = np.asarray(areas)
    sorted_idxs = np.argsort(-areas).tolist()
    keep_results = [keep_results[k]
                    for k in sorted_idxs] if len(keep_results) > 0 else []
    print(keep_results)
    print(count)
    f.write("the total number is :" + str(int(count)))
    f.close()
#cap = cv2.VideoCapture(0)
#while cap.isOpened():
#    ret, frame = cap.read()
#    if ret:
#        result = predictor.predict(frame)
#        vis_img = pdx.det.visualize(frame, result, threshold=0.6, save_dir=None)
#        cv2.imshow('Xiaoduxiong', vis_img)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    else:
#        break
#cap.release()
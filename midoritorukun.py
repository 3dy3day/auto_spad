import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

joint = os.path.join
current = os.getcwd()
inpath = 'input_img'
outpath = 'output_img'
path = joint(current, inpath)
load = os.listdir(path)


def imread(filename, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        decimg = cv2.imdecode(n, cv2.IMREAD_COLOR)
        return decimg
    except:
        print("ERROR!!")
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except:
        print("ERROR!!")
        return False


def main(name, img, before_img):
    export_name = name + '_result.png'
    hsvLower = np.array([50 / 360 * 179, 50, 20])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([150 / 360 * 179, 255, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(before_img, cv2.COLOR_RGB2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    # plt.imshow(img)
    # plt.title('img')
    # plt.show()
    # plt.close()
    # plt.imshow(before_img)
    # plt.title('before_img')
    # plt.show()
    # plt.close()
    # plt.imshow(hsv_mask)
    # plt.title('hsv_mask')
    # plt.show()
    # plt.close()
    result = cv2.bitwise_and(img, img, mask=hsv_mask)
    ave = cv2.mean(img, mask=hsv_mask)  # takes average of area of mask(leaf)
    # cv2.imshow('itle', img, mask=hsv_mask)
    # # plt.title('show with mask')
    # cv2.show()
    # cv2.waitKey(0)
    avve = cv2.mean(img)
    # print('with mask ', ave)
    # print('no mask ', avve)
    ave_color = [ave[0], ave[1], ave[2]]  # RGB

    # plt.imshow(result)
    # plt.show()
    # plt.close()
    # print('with mask ', ave)
    # print('no mask ', avve)

    # print('bgr with no mask ', cv2.mean(bgr))
    # print('bgr with mask ', cv2.mean(bgr, mask=hsv_mask))#takes area of mask
    # bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)# cv2 takes BGR so must be converted
    # cv2.imwrite(joint(outpath, export_name), bgr)
    return ave_color

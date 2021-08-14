import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os
from sklearn.linear_model import LinearRegression
import midoritorukun
import sample_tile_finder
import time
from tqdm import tqdm
import csv
rows = 50  # how many rows you need
cols = 50  # how many collums you need
tiles = rows*cols


def main(chunks, sorted_target, color, mode, debug, height, width):

    if mode == 0:
        if color == 'white':
            for i in range(len(chunks)):
                if sorted_target[i, 1] < 1000:  # finding white
                    id = sorted_target[i, 0].astype(np.int64)
                    dis = sorted_target[i, 1].astype(np.int64)
                    x = (id % rows)*(width//cols)
                    y = (id//rows)*(height//rows)
                    c_height, c_width, c_channel = np.shape(chunks[id])
                    white_dif = chunks[id][c_height//2, c_width//2]
                    if debug == True:
                        print('white', 'x:', x, 'y:', y, white_dif)
                        title = 'white ID: '+str(id)+' '+'Distance: ' + \
                            str(dis)+' Var: '+str(sorted_target[i, 2])
                        # pic=cv2.imread(path1)
                        # pic=cv2.rectangle(pic,(x,y),(x+(width//cols),y+(hight//rows)),(0,255,0),2)
                        plt.imshow(chunks[id])
                        plt.title(title)
                        plt.show()
                        plt.close()
                    return white_dif
            return [0]

        elif color == 'grey':
            for i in range(len(chunks)):
                # finding grey 2000 or
                if sorted_target[i, 2] < 20 and sorted_target[i, 1] < 2000:
                    id = sorted_target[i, 0].astype(np.int64)
                    dis = sorted_target[i, 1].astype(np.int64)
                    x = (id % rows)*(width//cols)
                    y = (id//rows)*(height//rows)
                    c_height, c_width, c_channel = np.shape(chunks[id])
                    grey_dif = chunks[id][c_height//2, c_width//2]
                    if debug == True:
                        print('grey', 'x:', x, 'y:', y, grey_dif)
                        title = 'grey ID: ' + \
                            str(id)+' '+'Distance: ' + str(dis) + \
                            ' Var: '+str(sorted_target[i, 2])
                        plt.imshow(chunks[id])
                        plt.title(title)
                        plt.show()
                        plt.close()
                    return grey_dif
            return [0]

        else:
            for i in range(len(chunks)):
                if sorted_target[i, 2] < 20 and sorted_target[i, 1] < 2000:  # fiding black
                    id = sorted_target[i, 0].astype(np.int64)
                    dis = sorted_target[i, 1].astype(np.int64)
                    x = (id % rows)*(width//cols)
                    y = (id//rows)*(height//rows)
                    c_height, c_width, c_channel = np.shape(chunks[id])
                    black_dif = chunks[id][c_height//2, c_width//2]
                    if debug == True:
                        print('black', 'x:', x, 'y:', y, black_dif)
                        title = 'Black ID: ' + \
                            str(id)+' '+'Distance: ' + str(dis) + \
                            ' Var: '+str(sorted_target[i, 2])
                        plt.imshow(chunks[id])
                        plt.title(title)
                        plt.show()
                        plt.close()
                    return black_dif
            return [0]

    elif mode == 1:
        if color == 'white':
            for i in range(len(chunks)):
                if sorted_target[i, 1] < 1000:  # finding white
                    id = sorted_target[i, 0].astype(np.int64)
                    dis = sorted_target[i, 1].astype(np.int64)
                    x = (id % rows)*(width//cols)
                    y = (id//rows)*(height//rows)
                    c_height, c_width, c_channel = np.shape(chunks[id])
                    white_dif = chunks[id][c_height//2, c_width//2]
                    if debug == True:
                        print('white', 'x:', x, 'y:', y, white_dif)
                        title = 'white ID: '+str(id)+' '+'Distance: ' + \
                            str(dis)+' Var: '+str(sorted_target[i, 2])
                        # pic=cv2.imread(path1)
                        # pic=cv2.rectangle(pic,(x,y),(x+(width//cols),y+(hight//rows)),(0,255,0),2)
                        plt.imshow(chunks[id])
                        plt.title(title)
                        plt.show()
                        plt.close()
                    return white_dif
            return [0]

        elif color == 'grey':
            for i in range(len(chunks)):
                # finding grey 2000 or
                if sorted_target[i, 2] < 20 and sorted_target[i, 1] < 7000:
                    id = sorted_target[i, 0].astype(np.int64)
                    dis = sorted_target[i, 1].astype(np.int64)
                    x = (id % rows)*(width//cols)
                    y = (id//rows)*(height//rows)
                    c_height, c_width, c_channel = np.shape(chunks[id])
                    grey_dif = chunks[id][c_height//2, c_width//2]
                    if debug == True:
                        print('grey', 'x:', x, 'y:', y, grey_dif)
                        title = 'grey ID: ' + \
                            str(id)+' '+'Distance: ' + str(dis) + \
                            ' Var: '+str(sorted_target[i, 2])
                        plt.imshow(chunks[id])
                        plt.title(title)
                        plt.show()
                        plt.close()
                    return grey_dif
            return [0]

        else:
            for i in range(len(chunks)):
                if sorted_target[i, 2] < 20 and sorted_target[i, 1] < 2000:  # fiding black
                    id = sorted_target[i, 0].astype(np.int64)
                    dis = sorted_target[i, 1].astype(np.int64)
                    x = (id % rows)*(width//cols)
                    y = (id//rows)*(height//rows)
                    c_height, c_width, c_channel = np.shape(chunks[id])
                    black_dif = chunks[id][c_height//2, c_width//2]
                    if debug == True:
                        print('black', 'x:', x, 'y:', y, black_dif)
                        title = 'Black ID: ' + \
                            str(id)+' '+'Distance: ' + str(dis) + \
                            ' Var: '+str(sorted_target[i, 2])
                        plt.imshow(chunks[id])
                        plt.title(title)
                        plt.show()
                        plt.close()
                    return black_dif
            return [0]

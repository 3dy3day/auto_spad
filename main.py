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
from natsort import natsorted

t1 = time.time()
joint = os.path.join
current = os.getcwd()
inpath = 'input_img'
learn_input = 'img_learn'
outpath = 'output_img'
profile_out = 'profile'
csv_learn = 'csv_learn'
answerimgpath = 'answer_img'
debug = False

# ipath = joint(current, inpath)
ipath = joint(current, answerimgpath)
lpath = joint(current, learn_input)
opath = joint(current, outpath)
profile_path = joint(current, profile_out)
csv_path = joint(current, csv_learn)
load = os.listdir(ipath)

# print(load)
# print(natsorted(load))

rows = 50  # how many rows you need
cols = 50  # how many collums you need
tiles = rows*cols

# colorsample c-lab(95*0*0, 48*0*0, 35*0*0)
# colorsample rgb(240,240,240  114,114,114  82,82,82)
# sample_rgb = [240, 114, 82]
# sample_rgb = [240, 160, 82]  # simple criterion
# sample_rgb = [240, 82, 30]  # only used to create profile


def euclid_dis(rgb, wsample):
    distance = (rgb[0]-wsample)**2 + (rgb[1]-wsample)**2 + (rgb[2]-wsample)**2
    return distance


def get_sample(img, path1, mode):
    height, width, channel = np.shape(img)
    chunks = []  # gets reseted in each call
    target_white = []
    target_grey = []
    target_black = []

    for row_img in np.array_split(img, rows, axis=0):  # spit image
        for chunk in np.array_split(row_img, cols, axis=1):
            chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        img_list = np.asarray(chunk)
        b_ave = chunk.T[2].flatten().mean()
        g_ave = chunk.T[1].flatten().mean()
        r_ave = chunk.T[0].flatten().mean()
        b_var = chunk.T[2].flatten().var()
        g_var = chunk.T[1].flatten().var()
        r_var = chunk.T[0].flatten().var()
        avergb = [r_ave, g_ave, b_ave]  # no need
        imgvar = [r_var, g_var, b_var]
        target_white.append(
            [i, euclid_dis(avergb, sample_rgb[0]), np.mean(imgvar)])
        target_grey.append(
            [i, euclid_dis(avergb, sample_rgb[1]), np.mean(imgvar)])
        target_black.append(
            [i, euclid_dis(avergb, sample_rgb[2]), np.mean(imgvar)])

    target_white = np.reshape(target_white, (tiles, 3))
    target_grey = np.reshape(target_grey, (tiles, 3))
    target_black = np.reshape(target_black, (tiles, 3))
    tar_w_sort = target_white[np.argsort(target_white[:, 2])]
    tar_g_sort = target_grey[np.argsort(target_grey[:, 2])]
    tar_b_sort = target_black[np.argsort(target_black[:, 1])]

    white_dif = sample_tile_finder.main(
        chunks, tar_w_sort, 'white', mode, debug, height, width)
    grey_dif = sample_tile_finder.main(
        chunks, tar_g_sort, 'grey', mode, debug, height, width)
    black_dif = sample_tile_finder.main(
        chunks, tar_b_sort, 'black', mode, debug, height, width)

    if white_dif[0] == 0 or grey_dif[0] == 0 or black_dif[0] == 0:
        print('color sample may be missing')
        return 0, 0, 0

    hoge = white_dif, grey_dif, black_dif
    return hoge


def get_formula(white_dif, grey_dif, black_dif):  # goes like red green blue
    red_formula = 0
    green_formula = 0
    blue_formula = 0
    formulas = [red_formula, green_formula, blue_formula]
    for k in range(3):  # k=0 red, k=1green
        x = [white_dif[k], grey_dif[k], black_dif[k]]
        x = np.array(x)
        y = [240, 114, 82]
        # y = [sample_rgb[0], sample_rgb[1], sample_rgb[2]]

        y = np.array(y)
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        reg = LinearRegression().fit(x, y)
        cmat = np.arange(0, 255, 1)
        predic = cmat*reg.coef_ + reg.intercept_
        predic = np.reshape(predic, (-1, 1))
        # plt.scatter(x, y)
        # plt.plot(cmat, predic, color='red')
        # plt.show()
        formulas[k] = [float(reg.coef_), float(reg.intercept_)]
    return formulas


def get_correct(img, formula):
    r, g, b = cv2.split(img)
    red = np.clip(r.astype(int)*formula[0]
                  [0]+formula[0][1], a_min=0, a_max=255)
    green = np.clip(g.astype(int)*formula[1]
                    [0]+formula[1][1], a_min=0, a_max=255)
    blue = np.clip(b.astype(int)*formula[2]
                   [0]+formula[2][1], a_min=0, a_max=255)
    red = red.astype(np.uint8)
    green = green.astype(np.uint8)
    blue = blue.astype(np.uint8)
    new_img = cv2.merge((red, green, blue))

    # plt.imshow(new_img)
    # plt.show()
    # plt.close()
    return new_img


def create_model(xmat, ymat, whichfile):
    if np.shape(xmat)[0] == np.shape(ymat)[0]:
        reg = LinearRegression().fit(xmat, ymat)
        print('slope:', reg.coef_)
        print('intercept:', reg.intercept_)
        score = reg.score(xmat, ymat)
        print('r2:', score)
        profile_name = whichfile + '.csv'

        with open(joint(profile_path, profile_name), 'w') as f:
            writer = csv.writer(f)
            # writer.writerow(reg.coef_)
            # writer.writerow(reg.intercept_)
            writer.writerow([reg.coef_[0], reg.coef_[
                            1], reg.coef_[2], reg.intercept_])

        # newprofile = open(joint(profile_path, profile_name), 'w')
        # newprofile.write(reg.coef_, reg.intercept_)
        # newprofile.close()

        predict = []
        for hoge in range(len(ymat)):
            fuga = xmat[hoge][0]*reg.coef_[0]+xmat[hoge][1] * \
                reg.coef_[1]+xmat[hoge][2]*reg.coef_[2]+reg.intercept_
            predict.append(fuga)
        save_result(ymat, predict, score)  # true, spredic, score(r2)
    else:
        print('The Number of Pictures and SPAD Value does not Match')

    return


def get_spad(pathtoprofile, rgb):
    data = np.loadtxt(pathtoprofile, delimiter=',')
    spad = rgb[0]*data[0] + rgb[1] * \
        data[1] + rgb[2]*data[2] + data[3]
    return spad


def cal_spad():
    print('Choose Profile')
    print(os.listdir(profile_path))
    whichprofile = 'soybean.csv'  # input()
    if whichprofile in os.listdir(profile_path):
        pathtoprofile = joint(profile_path, whichprofile)
    else:
        print('ERROR: There is no such file')
        return
    predic_spad = []  # use for calculating accuracy

    for j in natsorted(load):
        if j.endswith('.png') or j.endswith('.jpeg') or j.endswith('.jpg') or j.endswith('.JPG') or j.endswith('.JPEG'):
            path1 = joint(ipath, j)
            img = cv2.imread(path1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print('processing ', j)
            white_dif, grey_dif, black_dif = get_sample(img, path1, 1)
            if type(white_dif) == np.ndarray:
                formula = get_formula(white_dif, grey_dif, black_dif)
                cor_img = get_correct(img, formula)
                # plt.imshow(cor_img)
                # plt.show()
                # plt.close()
                green_ave_color = midoritorukun.main(j, cor_img, img)
                spad = get_spad(pathtoprofile, green_ave_color)
                # spad = np.round(spad, 2)
                predic_spad.append(spad)  # use for calculating accuracy
                bake_result(j, img, spad)

                print('spad value of', j, 'is:', spad)
            else:
                print('ERROR: no color sample found')
                return

    calculate_r(predic_spad)
    return


def calculate_r(predic_spad):
    from sklearn.metrics import r2_score
    anspath = joint(current, 'answer_csv')
    data = np.loadtxt(joint(anspath, 'soybean_answer_1-5.csv'),
                      delimiter=',', skiprows=1, usecols=range(1, 3))  # pick which answer sheet is being used
    true_spad = data[:, 1]
    # true_spad = np.reshape(true_spad, (1, -1))

    score = r2_score(true_spad, predic_spad)
    save_result(true_spad, predic_spad, score)
    print('R2:', score)
    return


def save_result(true_spad, predic_spad, score):
    # with open(joint(profile_path, profile_name), 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(true_spad)
    #     writer.writerow(predic_spad)
    id = np.arange(len(predic_spad))
    p1 = plt.plot(id, true_spad, color='red')
    p2 = plt.plot(id, predic_spad, color='blue')
    plt.legend((p1[0], p2[0]), ("true spad", "predic spad"), loc=2)
    plt.title(score)
    plt.show()
    plt.close()


def bake_result(name, img, spad):
    export_name = os.path.splitext(name)[0]+'_result.png'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img, str(spad), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2.5,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(joint(opath, export_name), img)
    return


def create_new_profile():
    print('Choose The File That Contains Images to Learn (This may take a while)')
    print(os.listdir(lpath))
    whichfile = input()
    pathtoimg = joint(lpath, whichfile)
    # print(os.listdir(filetoimg))
    print('Choose The File That Contains CSV to Learn')
    print(os.listdir(csv_path))
    whichcsv = input()
    pathtocsv = joint(csv_path, whichcsv)
    csvload = np.loadtxt(str(pathtocsv), dtype='float',
                         delimiter=',', skiprows=1)
    spad = csvload[:, 1]
    # print(spad)
    xmat = []
    ymat = spad

    for j in natsorted(os.listdir(pathtoimg)):
        # for _ in tqdm(range(len(os.listdir(pathtoimg))), desc="Creating New Profile"):
        if j.endswith('.png') or j.endswith('.jpeg') or j.endswith('.jpg') or j.endswith('.JPG') or j.endswith('.JPEG'):
            path1 = joint(pathtoimg, j)
            img = cv2.imread(path1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print('processing ', j)
            white_dif, grey_dif, black_dif = get_sample(img, path1, 0)
            if type(white_dif) == np.ndarray:
                formula = get_formula(white_dif, grey_dif, black_dif)
                cor_img = get_correct(img, formula)
                # plt.imshow(cor_img)
                # plt.show()
                # plt.close()
                green_ave_color = midoritorukun.main(j, cor_img, img)
                xmat.append(green_ave_color)
                # print('\n average green: ', green_ave_color, '\n')
                # print(np.shape(xmat), np.shape(ymat))
            else:
                print('ERROR: no color sample found in file:', j)
                return
        else:
            print('Invalid File')
            return
    create_model(xmat, ymat, whichfile)
    return


if __name__ == '__main__':
    print('Select Mode: Create New Profile[0], Calculate SPAD[1]')
    ans = int(input())
    if ans == 0:
        sample_rgb = [240, 82, 30]  # only used to create profile
        create_new_profile()
    elif ans == 1:
        sample_rgb = [240, 160, 82]  # simple criterion
        cal_spad()
    else:
        print('Invalid Response')
t2 = time.time()
elapsed_time = t2-t1
print('\n', "Run Time:", np.round(elapsed_time, 2), 'sec')
print("Program End")

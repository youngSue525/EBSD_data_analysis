import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

from sklearn.cluster import KMeans
import math
from scipy.stats import norm
import cv2


def color_points(x, y, radius, num):
    t = np.random.random(size=num) * 2 * np.pi - np.pi
    dx = np.cos(t)
    dy = np.sin(t)

    xy_list = []
    for i in range(num):
        len = np.sqrt(np.random.random()) * radius
        x1 = float(dx[i] * len) + x
        y1 = float(dy[i] * len) + y
        xy_list.append((x1, y1))

    return xy_list


def color_list_bygrain(path, width, height):
    img001 = cv2.imread(path)
    img001 = cv2.resize(img001, (width, height))
    img001 = cv2.cvtColor(img001, cv2.COLOR_BGR2RGB)

    b, g, r = cv2.split(img001)
    b = np.reshape(b, (height * width, 1))
    b_list = []
    g = np.reshape(g, (height * width, 1))
    g_list = []
    r = np.reshape(r, (height * width, 1))
    r_list = []

    for i in range(height * width):
        n = 255  # 255

        b0 = float(b[i]) / n
        b0 = round(b0, 2)
        b0 = int(n * b0)
        b_list.append(b0)

        g0 = float(g[i]) / n
        g0 = round(g0, 2)
        g0 = int(n * g0)
        g_list.append(g0)

        r0 = float(r[i]) / n
        r0 = round(r0, 2)
        r0 = int(n * r0)
        r_list.append(r0)

    b_array = np.array(b_list)
    g_array = np.array(g_list)
    r_array = np.array(r_list)

    bgr = cv2.merge([b_array, g_array, r_array])
    image = np.reshape(bgr, (height * width, 3))
    image_list = image.tolist()[::1]  # 100
    image_list = sorted(image_list, key=lambda x: x[0])
    image_list_new = image_list.copy()

    for i in range(len(image_list) - 2, -1, -1):

        b = image_list[i + 1][0]
        g = image_list[i + 1][1]
        r = image_list[i + 1][2]

        bb = image_list[i][0]
        gg = image_list[i][1]
        rr = image_list[i][2]

        min = 0.9  # 设置 rgb 相同的阈值， 放宽导致number减小
        max = 1.1

        if bb == 0 and gg == 0 and rr == 0:
            image_list_new.remove(image_list_new[i])

        elif min * b <= bb <= max * b and min * g <= gg <= max * g and min * r <= rr <= max * r:
            image_list[i] = image_list[i + 1]
            image_list_new.remove(image_list_new[i])

    size = []
    image_list_new2 = []
    size_new = []

    for i in image_list_new:
        grain_size = image_list.count(i)

        if grain_size >= 10:
            image_list_new2.append(i)
            size.append(grain_size)
            size1 = 1 * grain_size      # 像素 * 比例 = 尺寸
            size_new.append(size1)


    # print('像素总数：', len(image_list), end='\r')
    # print('不同像素种类：', len(image_list_new2))

    # print('像素：', (image_list))
    print('不同像素：', (image_list_new2))        # need

    print('各个晶粒包含像素数目：', size)       # need
    # print('各个晶粒尺寸：', size_new)
    print('晶粒数目：', len(size))

    return image_list_new2, size

def sep_color(img_arr):
    # width = img_arr.shape[1]
    # height = img_arr.shape[0]
    # img_arr = np.reshape(img_arr, (width * height, 3))
    r , g , b = img_arr[...,0], img_arr[...,1], img_arr[...,2]

    r_list = []
    g_list = []
    b_list = []
    for i in range(img_arr.shape[0]):
        if r[i] == 255 and g[i] == 255 and b[i] == 255:
            pass
        elif r[i] == 255:
            r_list.append(img_arr[i])
        elif g[i] == 255:
            g_list.append(img_arr[i])
        elif b[i] == 255:
            b_list.append(img_arr[i])
        else:
            pass
    return r_list, g_list, b_list


def color_list_new(main, mu, sigma, num):
    r = main[0]
    g = main[1]
    b = main[2]
    color_list = []

    if r == 255:
        distance = np.random.normal(mu, sigma, num)
        for i in range(num):
            gb_list = color_points(g, b, distance[i], 1)
            # print(gb_list[0])
            rgb = (255, gb_list[0][0], gb_list[0][1])
            color_list.append(rgb)

    return color_list



if __name__ == "__main__":

    a = color_list_new([255.,138.86325803,133.46254459], 70.1231012819373, 34.45736401882563, 100)
    print(a)
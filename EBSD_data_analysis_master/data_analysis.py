import numpy as np
import random
import matplotlib.pyplot as plt
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

def sep_color(img_arr):
    width = img_arr.shape[1]
    height = img_arr.shape[0]
    img_arr = np.reshape(img_arr, (width * height, 3))
    r , g , b = img_arr[...,0], img_arr[...,1], img_arr[...,2]

    r_list = []
    g_list = []
    b_list = []
    for i in range(width * height):
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


def cluster_3parts(path):

##########  读取原图的配色信息  ################

    img = plt.imread(path)

    # print(img.shape)
    r_list, g_list, b_list = sep_color(img)
    # print(len(r_list) / (len(r_list) +len(b_list)+len(g_list)))
    # print(len(g_list) / (len(r_list) +len(b_list)+len(g_list)))
    # print(len(b_list) / (len(r_list) +len(b_list)+len(g_list)))
    # print((len(r_list) +len(b_list)+len(g_list)))

    cluster = KMeans(n_clusters = 1, random_state = 0)
    cluster_r = cluster.fit(r_list)
    # y_pred = cluster.labels_
    centroid_r = cluster_r.cluster_centers_
    print('R 部分的聚类中心是：',centroid_r)

    cluster_g = cluster.fit(g_list)
    centroid_g = cluster_g.cluster_centers_
    print('G 部分的聚类中心是：',centroid_g)

    cluster_b = cluster.fit(b_list)
    centroid_b = cluster_b.cluster_centers_
    print('B 部分的聚类中心是：',centroid_b)


    distance_r = []
    for i in r_list:
        g = i[1]
        b = i[2]
        distance = math.sqrt((g-centroid_r[0][1])**2 + (b-centroid_r[0][2])**2)
        distance_r.append(distance)

    distance_g = []
    for i in g_list:
        r = i[0]
        b = i[2]
        distance = math.sqrt((r-centroid_r[0][1])**2 + (b-centroid_r[0][2])**2)
        distance_g.append(distance)

    distance_b = []
    for i in b_list:
        g = i[1]
        r = i[0]
        distance = math.sqrt((g-centroid_r[0][1])**2 + (r-centroid_r[0][2])**2)
        distance_b.append(distance)

    print('R 部分的数量：',len(distance_r))
    print('G 部分的数量：',len(distance_g))
    print('B 部分的数量：',len(distance_b))
    # print(max(distance_r))
    # print(min(distance_r))

    mu1 =np.mean(distance_r) #计算均值
    sigma1 =np.std(distance_r) #计算方差
    print(mu1, sigma1)
    mu2 = np.mean(distance_g)
    sigma2 = np.std(distance_g)
    print(mu2, sigma2)
    mu3 = np.mean(distance_b)
    sigma3 = np.std(distance_b)
    print(mu3, sigma3)
    
    return centroid_r, mu1, sigma1, centroid_g, mu2, sigma2, centroid_b, mu3, sigma3




    # figure, axs=plt.subplots(3,2, figsize = (10,8), constrained_layout=True)
    #
    #
    #
    #
    # n_r1, bins_r1, patches_r1 = axs[0][0].hist(distance_r, 500, density=True, facecolor='r', alpha=0.5, log=False)           # 第二个参数代表柱状图的数量
    # y_r1 = norm.pdf(bins_r1, mu1, sigma1)
    # axs[0][0].plot(bins_r1, y_r1, c='black', linestyle='--') #绘制y的曲线
    # axs[0][0].set_xlim([min(distance_r), max(distance_r)])
    # axs[0][0].set_ylim([0, 0.025])
    #
    # gen_y_r1 = np.random.normal(mu1, sigma1, len(distance_r))
    # print(gen_y_r1)
    # n_r2, bins_r2, patches_r2 = axs[0][1].hist(gen_y_r1, 500, range=(min(distance_r), max(distance_r)),  density=True, facecolor='r', alpha=0.5, log=False)
    # y_r2 = norm.pdf(bins_r2, mu1, sigma1)
    # axs[0][1].plot(bins_r2, y_r2, c='black', linestyle='--') #绘制y的曲线
    # axs[0][1].set_xlim([min(distance_r), max(distance_r)])
    # axs[0][1].set_ylim([0, 0.025])
    #
    #
    #
    # n_g1, bins_g1, patches_g1 = axs[1][0].hist(distance_g, 500, density=True, facecolor='g', alpha=0.5, log=False)
    # y_g1 = norm.pdf(bins_g1, mu2, sigma2)
    # axs[1][0].plot(bins_g1, y_g1, c='black', linestyle='--')  # 绘制y的曲线
    # axs[1][0].set_xlim([min(distance_g), max(distance_g)])
    # axs[1][0].set_ylim([0, 0.025])
    #
    # gen_y_g1 = np.random.normal(mu2, sigma2, len(distance_g))
    # n_g2, bins_g2, patches_g2 = axs[1][1].hist(gen_y_g1, 500, range=(min(distance_g), max(distance_g)), density=True, facecolor='g',
    #                                      alpha=0.5, log=False)
    # y_g2 = norm.pdf(bins_g2, mu2, sigma2)
    # axs[1][1].plot(bins_g2, y_g2, c='black', linestyle='--')  # 绘制y的曲线
    # axs[1][1].set_xlim([min(distance_g), max(distance_g)])
    # axs[1][1].set_ylim([0, 0.025])
    #
    # n_b1, bins_b1, patches_b1 = axs[2][0].hist(distance_b, 500, density=True, facecolor='b', alpha=0.5, log=False)
    # y_b1 = norm.pdf(bins_b1, mu3, sigma3)
    # axs[2][0].plot(bins_b1, y_b1, c='black', linestyle='--')  # 绘制y的曲线
    # axs[2][0].set_xlim([min(distance_b), max(distance_b)])
    # axs[2][0].set_ylim([0, 0.025])
    #
    # gen_y_b1 = np.random.normal(mu3, sigma3, len(distance_b))
    # n_b2, bins_b2, patches_b2 = axs[2][1].hist(gen_y_b1, 500, range=(min(distance_b), max(distance_b)), density=True, facecolor='b',
    #                                      alpha=0.5, log=False)
    # y_b2 = norm.pdf(bins_b2, mu3, sigma3)
    # axs[2][1].plot(bins_b2, y_b2, c='black', linestyle='--')  # 绘制y的曲线
    # axs[2][1].set_xlim([min(distance_b), max(distance_b)])
    # axs[2][1].set_ylim([0, 0.025])
    #
    # plt.show()

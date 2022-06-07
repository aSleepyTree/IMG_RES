import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import copy
import time


def norm(img1, img2):
    """
    计算范数
    """
    x = cv2.absdiff(img1, img2)
    x_norm = np.linalg.norm(x, ord=None, axis=None, keepdims=False)
    return x_norm


def noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6]):
    """
    根据题目要求生成受损图片
    """
    # 受损图片初始化None
    noise_img = None
    noise_img = np.copy(img)
    rows = img.shape[0]
    cols = img.shape[1]
    for row in range(rows):
        # 实现图像损坏，对RGB通道分别以0.8,0.4,0.6的比率添加噪声，下先确定噪点位置
        list_R = random.sample(range(cols), int(cols * noise_ratio[0]))
        list_G = random.sample(range(cols), int(cols * noise_ratio[1]))
        list_B = random.sample(range(cols), int(cols * noise_ratio[2]))
        # 对每一行分别损坏三个通道的像素点
        for i in range(int(cols * noise_ratio[0])):
            noise_img[row][list_R[i]][0] = 0.0
        for i in range(int(cols * noise_ratio[1])):
            noise_img[row][list_G[i]][1] = 0.0
        for i in range(int(cols * noise_ratio[2])):
            noise_img[row][list_B[i]][2] = 0.0

    return noise_img


def restore_image(noise_img, size=3):
    """
    使用区域二元线性回归模型进行图像恢复。
    noise_img: 受损的图像
    size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 3
    函数进行了均值去噪，均值去噪+线性回归去噪以及线性回归去噪
    average_img,res_img及res_img1按顺序分别为上述方法的返回值
    """
    # 读取受损图像
    res_img = np.copy(noise_img)
    noise_mask = np.copy(noise_img)
    res_img1 = np.copy(res_img)
    # 下进行图像恢复
    rows = res_img.shape[0]  # 行数
    cols = res_img.shape[1]  # 列数
    # -----------------------------线性回归开始---------------------
    Reg = LinearRegression()
    k = 0
    for chan in range(3):  # 处理各个通道
        for row in range(rows):
            for col in range(cols):
                k+=1
                if noise_mask[row, col, chan] != 0:
                    continue
                row_min = max(row - size, 0)
                row_max = min(row + size + 1, rows)
                col_min = max(col - size, 0)
                col_max = min(col + size + 1, cols)
                x_train = []
                y_train = []
                for x in range(row_min, row_max):
                    for y in range(col_min, col_max):
                        if x == row and y == col:
                            continue
                        if noise_img[x][y][chan] == 0:
                            continue
                        x_train.append([x, y])
                        y_train.append(res_img1[x][y][chan])
                if x_train == []:
                    size_larger = size
                    number = 0
                    while number == 0:  # 循环直到找到未受损点
                        size_larger += 1  # 扩大搜索范围
                        for x in range(
                            max(row - size_larger, 0), min(row + size_larger + 1, rows)
                        ):
                            for y in range(
                                max(col - size_larger, 0),
                                min(col + size_larger + 1, cols),
                            ):
                                if noise_mask[x][y][chan] != 0:
                                    x_train.append([x, y])
                                    y_train.append(res_img1[x][y][chan])
                                    number = 1
                
                    
                if col == 0:
                    print("now", (rows * chan + row), "of", 3 * rows)
                Reg.fit(x_train, y_train)
                res_img1[row, col, chan] = int(Reg.predict([[row, col]])[0])
                if k ==1:
                    print(x_train)
                    print()
                    print(y_train)
                    print()
                    print(res_img1[row, col, chan])

    # ---------------------------------------------------------------
    return res_img1

img = cv2.imread("test_s.jpg")
noise_image = noise_mask_image(img)
cv2.imwrite("test_noise.jpg", noise_image)
res_image1 = restore_image(noise_image)
cv2.imshow("res-pic", res_image1)
cv2.imwrite("res.jpg", res_image1)
cv2.imshow("noise", noise_image)
print("norm of noise and sourse:", norm(img, noise_image))
print("norm of LinearRegression and sourse:", norm(img, res_image1))

cv2.waitKey(0)

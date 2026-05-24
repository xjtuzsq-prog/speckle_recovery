import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
#定义自相关函数，调用后返回图像自相关且归一化后的图像
def zxg(img0):
    img = np.fft.ifftshift(img0)
    f = np.fft.fft2(img)
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft2(power).real
    autocorr_shifted = np.fft.fftshift(autocorr)
    autocorr_norm = (autocorr_shifted - np.min(autocorr_shifted)) / (
        np.max(autocorr_shifted) - np.min(autocorr_shifted)
    )
    return autocorr_norm
#调用相位恢复算法，调用后直接返回相位恢复后的图像
def phase_recovery(img, beta_start, beta_step, beta_end, N_iter, initial_guess):
    img1 = np.float32(img)
    g1 = np.float32(initial_guess)
    ii = 0
    BETAS = np.arange(beta_start, beta_end + beta_step, beta_step)
    for ibeta in range(len(BETAS)):
        beta = BETAS[ibeta]
        for iter in range(N_iter):
            ii += 1
            G_uv = np.fft.fft2(g1)
            g1_tag = np.real(np.fft.ifft2(img1 * G_uv / np.abs(G_uv)))
            g1 = g1_tag * (g1_tag >= 0) + (g1_tag < 0) * (g1 - beta * g1_tag)
    for iter in range(N_iter):
        ii += 1
        G_uv = np.fft.fft2(g1)
        g1_tag = np.real(np.fft.ifft2(img1 * G_uv / np.abs(G_uv)))
        g1 = g1_tag * (g1_tag >= 0)
    return g1
#调用“原图除以低通滤波”整体算法，同时该函数中包含了对原始图像的预处理
def high_pass_filter(I0,filter_length):
    I1 = np.array(I0)
    I1 = np.fliplr(I1)#反转图像得到正向的图

    N1, N2 = I1.shape
    T = np.zeros((N1, N2))
    center_row = N1 // 2
    center_col = N2 // 2
    row_start = max(0, center_row - filter_length)
    row_end = min(N1, center_row + filter_length + 1)
    col_start = max(0, center_col - filter_length)
    col_end = min(N2, center_col + filter_length + 1)
    T[row_start:row_end, col_start:col_end] = 1

    I2 = np.fft.fftshift(np.fft.fft2(I1))
    I3 = I2 * T
    I4 = np.abs(np.fft.ifft2(np.fft.ifftshift(I3)))
    I = I1 / I4#除以低通滤波后的图，使得高频部分更加清晰明显
    return I,N1,N2,center_row,center_col
#同态滤波算法
def Homo_filter(I,N1,N2,center_row, center_col):
    I_log = np.log(I + 1e-6)  # 对数变换（避免除零，加极小值）
    I_fft = np.fft.fft2(I_log)
    I_fft_shift = np.fft.fftshift(I_fft)

    H = np.ones_like(I_fft_shift)  # 构建同态滤波器（高斯型带通）
    D0 = 8  # 截止频率（默认值，可调整）
    gamma_L, gamma_H = 0.6, 1.8  # 低频抑制/高频增强系数
    c = 1
    for i in range(N1):
        for j in range(N2):
            D = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            H[i, j] = (gamma_H - gamma_L) * (1 - np.exp((-1 * c * (D ** 2)) / (D0 ** 2))) + gamma_L  # 高斯型传递函数

    I_filtered_fft = I_fft_shift * H
    I_filtered = np.fft.ifft2(np.fft.ifftshift(I_filtered_fft))
    I_homo = np.exp(np.real(I_filtered))  # 指数变换还原，取实部避免复数
    I_norm = (I_homo - np.min(I_homo)) / (np.max(I_homo) - np.min(I_homo))#归一化
    return I_norm
#处理方案1：在处理过程中加入同态滤波
def process_homo(I0, filter_length, pic_length):
    I,N1,N2,center_row,center_col=high_pass_filter(I0, filter_length)
    I=Homo_filter(I,N1,N2,center_row, center_col)
    zxgI = zxg(I)

    img_piece = zxgI[center_row - pic_length:center_row + pic_length, center_col - pic_length:center_col + pic_length]

    Image.fromarray(img_piece * 255).show()
    fixed_amp = np.sqrt(np.abs(np.fft.fft2(img_piece)))
    re = phase_recovery(fixed_amp, 2, -0.02, 0, 120, np.random.rand(img_piece.shape[0], img_piece.shape[1]))

    re_norm = (re - np.min(re)) / (np.max(re) - np.min(re))

    # 绘制带比例尺的结果（无白边显示）
    re_blc = re_norm.copy()
    height, width = re_blc.shape
    scalebar_length, scalebar_thickness = 20, 1
    margin, proj_length = 5, 1

    y_pos = height - margin - scalebar_thickness
    x_start, x_end = margin, margin + scalebar_length
    re_blc[y_pos:y_pos + scalebar_thickness, x_start:x_end] = re_blc.max()
    re_blc[y_pos - scalebar_thickness:y_pos + 2 * scalebar_thickness, x_start:x_start + proj_length] = re_blc.max()
    re_blc[y_pos - scalebar_thickness:y_pos + 2 * scalebar_thickness, x_end - proj_length:x_end] = re_blc.max()

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)  # 尺寸匹配图像像素
    ax = fig.add_axes([0, 0, 1, 1])  # 图像占满整个画布
    ax.imshow(re_blc * 255, cmap='hot')
    ax.axis('off')  # 关闭所有坐标轴装饰
    plt.show()

    return re_blc
#处理方案2：相比于处理方案1，减少了同态滤波处理，其他都不变。
def process(I0, filter_length, pic_length):
    I,N1,N2,center_row,center_col=high_pass_filter(I0, filter_length)
    zxgI = zxg(I)
    img_piece = zxgI[center_row - pic_length:center_row + pic_length, center_col - pic_length:center_col + pic_length]
    center_line = img_piece[img_piece.shape[0] // 2, :]  # 提取水平中心线强度分布

    Image.fromarray(img_piece * 255).show()
    fixed_amp = np.sqrt(np.abs(np.fft.fft2(img_piece)))
    re = phase_recovery(fixed_amp, 2, -0.02, 0, 120, np.random.rand(img_piece.shape[0], img_piece.shape[1]))

    re_norm = (re - np.min(re)) / (np.max(re) - np.min(re))

    # 绘制带比例尺的结果（无白边显示）
    re_blc = re_norm.copy()
    height, width = re_blc.shape
    scalebar_length, scalebar_thickness = 20, 1
    margin, proj_length = 5, 1

    y_pos = height - margin - scalebar_thickness
    x_start, x_end = margin, margin + scalebar_length
    re_blc[y_pos:y_pos + scalebar_thickness, x_start:x_end] = re_blc.max()
    re_blc[y_pos - scalebar_thickness:y_pos + 2 * scalebar_thickness, x_start:x_start + proj_length] = re_blc.max()
    re_blc[y_pos - scalebar_thickness:y_pos + 2 * scalebar_thickness, x_end - proj_length:x_end] = re_blc.max()

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)  # 尺寸匹配图像像素
    ax = fig.add_axes([0, 0, 1, 1])  # 图像占满整个画布
    ax.imshow(re_blc * 255, cmap='hot')
    ax.axis('off')  # 关闭所有坐标轴装饰
    plt.show()
    return center_line

I0 = Image.open(r"C:\ProgramData\Galaxy\userdata\ImagesAndVideos\Pic_20251110192438510.bmp")
I0=I0.convert("L")
filter_length = 3
pic_length = 80
process(I0, filter_length, pic_length)
process_homo(I0, filter_length, pic_length)

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy.typing as npt
img1=np.zeros((1024,1024),dtype=np.uint8)
center_x=img1.shape[1]//2
center_y=img1.shape[0]//2

offset_right = 0  # 向右偏移100像素
center_x = center_x + offset_right
center_y=center_y + offset_right
horiz_width = 60   # 水平条变长
horiz_height = 10   # 水平条变细

vert_width = 10     # 垂直条变细
vert_height = 60   # 垂直条变长

start_x_horiz = center_x - horiz_width // 2
start_y_horiz = center_y - horiz_height // 2
end_x_horiz = center_x + horiz_width // 2
end_y_horiz = center_y + horiz_height // 2

start_x_vert = center_x - vert_width // 2
start_y_vert = center_y - vert_height // 2
end_x_vert = center_x + vert_width // 2
end_y_vert = center_y + vert_height // 2

# 绘制十字
img1[start_y_horiz:end_y_horiz, start_x_horiz:end_x_horiz] = 1
img1[start_y_vert:end_y_vert, start_x_vert:end_x_vert] = 1
img11=img1*255
np.random.seed(40)
img2 = np.random.uniform(0, 255, size=(1024, 1024))  # 均匀分布，范围 0~255
img22 = img2.astype(np.uint8)
##Image.fromarray(img1).show()
plt.imshow(img11,cmap='hot')
#plt.colorbar()  # 可选：显示颜色刻度
plt.xticks([])  # 清除x轴刻度
plt.yticks([])  # 清除y轴刻度
plt.show()
##Image.fromarray(img2).show()
plt.imshow(img22,cmap='hot')
#plt.colorbar()  # 可选：显示颜色刻度
plt.xticks([])  # 清除x轴刻度
plt.yticks([])  # 清除y轴刻度
plt.show()

def self_convolve2d(img1,img2):
    fft_img1 = np.fft.fft2(img1)
    fft_img2 = np.fft.fft2(img2)
    ff_result=fft_img1*fft_img2
    f_result=np.fft.ifft2(ff_result)
    result=np.real(np.fft.fftshift(f_result))
    return result
I=self_convolve2d(img1,img2)
I_norm=(I-np.min(I))/(np.max(I)-np.min(I))
#Image.fromarray(I_norm*255).show()
plt.imshow(I_norm*255,cmap='hot')
#plt.colorbar()  # 可选：显示颜色刻度
plt.xticks([])  # 清除x轴刻度
plt.yticks([])  # 清除y轴刻度
plt.show()
def zxg(img0):
    img = np.fft.ifftshift(img0)
    f = np.fft.fft2(img)
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft2(power).real
    autocorr_shifted = np.fft.fftshift(autocorr)
    autocorr_norm = (autocorr_shifted - np.min(autocorr_shifted)) / (
            np.max(autocorr_shifted) - np.min(autocorr_shifted))
    return autocorr_norm
zxgI=zxg(I_norm)
zxgimg1=zxg(img1)
img_piece1=zxgimg1[center_x-128:center_x+128,center_y-128:center_y+128]
#截取中心有效的区域数据，大概128*128？
img_piece=zxgI[center_x-128:center_x+128,center_y-128:center_y+128]
##Image.fromarray(img_piece*255).show()
plt.imshow(img_piece*255,cmap='hot')
#plt.colorbar()  # 可选：显示颜色刻度
plt.xticks([])  # 清除x轴刻度
plt.yticks([])  # 清除y轴刻度
plt.show()
plt.imshow(img_piece1*255,cmap='hot')
#plt.colorbar()  # 可选：显示颜色刻度
plt.xticks([])  # 清除x轴刻度
plt.yticks([])  # 清除y轴刻度
plt.show()
def phase_recovery(img,beta_start,beta_step,beta_end,N_iter,initial_guess):
    img1=np.float32(img)
    g1=np.float32(initial_guess)
    ii=0
    BETAS=np.arange(beta_start,beta_end+beta_step,beta_step)
    for ibeta in range(len(BETAS)):
        beta=BETAS[ibeta]
        for iter in range(N_iter):
            ii+=1
            G_uv=np.fft.fft2(g1)
            g1_tag = np.real(np.fft.ifft2(img1 * G_uv / np.abs(G_uv)))
            g1 = g1_tag * (g1_tag >= 0) + (g1_tag < 0) * (g1 - beta * g1_tag)
    for iter in range(N_iter):
            ii += 1
                # 傅里叶变换到频域
            G_uv = np.fft.fft2(g1)
                # 应用幅度约束并逆变换回空间域
            g1_tag = np.real(np.fft.ifft2(img1 * G_uv / np.abs(G_uv)))
                # 应用非负约束（仅保留非负值）
            g1 = g1_tag * (g1_tag >= 0)
    return g1
def phase_recovery(img,beta_start,beta_step,beta_end,N_iter,initial_guess):
    img1=np.float32(img)
    g1=np.float32(initial_guess)
    ii=0
    BETAS=np.arange(beta_start,beta_end+beta_step,beta_step)
    for ibeta in range(len(BETAS)):
        beta=BETAS[ibeta]
        for iter in range(N_iter):
            ii+=1
            G_uv=np.fft.fft2(g1)
            g1_tag = np.real(np.fft.ifft2(img1 * G_uv / np.abs(G_uv)))
            g1 = g1_tag * (g1_tag >= 0) + (g1_tag < 0) * (g1 - beta * g1_tag)
    for iter in range(N_iter):
            ii += 1
                # 傅里叶变换到频域
            G_uv = np.fft.fft2(g1)
                # 应用幅度约束并逆变换回空间域
            g1_tag = np.real(np.fft.ifft2(img1 * G_uv / np.abs(G_uv)))
                # 应用非负约束（仅保留非负值）
            g1 = g1_tag * (g1_tag >= 0)
    return g1
fixed_amp=np.sqrt(np.abs(np.fft.fft2(img_piece)))
re=phase_recovery(fixed_amp,2,-0.02,0,120,np.random.rand(img_piece.shape[0],img_piece.shape[1]))
##Image.fromarray(re*2550).show()
re_norm=(re-np.min(re))/(np.max(re)-np.min(re))
plt.imshow(re_norm*255, cmap='hot')
#plt.colorbar()  # 可选：显示颜色刻度
plt.xticks([])  # 清除x轴刻度
plt.yticks([])  # 清除y轴刻度
plt.show()

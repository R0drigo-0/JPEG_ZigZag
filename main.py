import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct


def dct_zigzag(bloc):
    M = dct(dct(bloc, axis=0, norm="ortho"), axis=1, norm="ortho")

    zigzag = []
    for i in range(8):
        for j in range(i + 1):
            zigzag.append(M[i][j])
        for j in range(i + 1, 8):
            zigzag.append(M[j][i])

    return zigzag


img = cv2.imread("image8.png")
img_YCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

uniform_bloc = img_YCC[0:8, 0:8, :]
color_change_bloc = img_YCC[0:8, 524:532]

uniform_Y = uniform_bloc[:, :, 0]
uniform_Cb = uniform_bloc[:, :, 1]
uniform_Cr = uniform_bloc[:, :, 2]

color_change_Y = color_change_bloc[:, :, 0]
color_change_Cb = color_change_bloc[:, :, 1]
color_change_Cr = color_change_bloc[:, :, 2]

dct_uniform_y = dct_zigzag(uniform_Y)
dct_uniform_Cb = dct_zigzag(uniform_Cb)
dct_uniform_Cr = dct_zigzag(uniform_Cr)

dct_color_change_y = dct_zigzag(color_change_Y)
dct_color_change_Cb = dct_zigzag(color_change_Cb)
dct_color_change_Cr = dct_zigzag(color_change_Cr)

matriu_quantitzacio_luminancia = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

matriu_quantitzacio_chrominancia = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)

q_luminancia_uniform = np.round(
    dct_uniform_y / matriu_quantitzacio_luminancia.flatten()
)

q_luminancia_color_change = np.round(
    dct_color_change_y / matriu_quantitzacio_luminancia.flatten()
)

print("Bloque uniforme:")
print(q_luminancia_uniform)
print()
print("Bloque cambio de color:")
print(q_luminancia_color_change)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(color_change_Y, cmap="gray")
axes[0].set_title("Y Channel")
axes[1].imshow(color_change_Cb, cmap="gray")
axes[1].set_title("Cb Channel")
axes[2].imshow(color_change_Cr, cmap="gray")
axes[2].set_title("Cr Channel")
fig.suptitle("Color Channels Color Change Image", fontsize=12)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(uniform_Y, cmap="gray")
axes[0].set_title("Y Channel")
axes[1].imshow(uniform_Cb, cmap="gray")
axes[1].set_title("Cb Channel")
axes[2].imshow(uniform_Cr, cmap="gray")
axes[2].set_title("Cr Channel")
fig.suptitle("Color Channels Uniform Image", fontsize=12)
plt.tight_layout()
plt.show()

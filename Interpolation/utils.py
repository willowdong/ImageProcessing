import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse


# RBF functions
def rbf_functions(r, epsilon=1, func="gaussian"):
    if func == "gaussian":
        return np.exp(-(epsilon * r) ** 2/2)
    elif func == "multiquadric":
        return np.sqrt(1 + 0.3*(epsilon * r) ** 2)
    elif func == "inverse_quad":            #no
        return 1 / (1 + (epsilon * r) ** 2)
    elif func == "tps":
        return r * r * np.log(r+1)
    elif func == "inverse_multiquad":
        return 1 / np.sqrt(1 + (epsilon * r) ** 2)
    elif func == "linear":
        return r
    else:
        print("No such rbf-function")
        exit(-1)


# print(rbf_functions(0))

def distance(x1, y1, x2, y2, strategy="Euclid"):
    if (strategy == "Euclid"):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    elif (strategy == "Manhattan"):
        return abs(x1 - x2) + abs(y1 - y2)
    else:
        print("No such distance strategy")
        exit(-1)


# test=np.ones([2,2])
# print(inverse_multiquad(test))

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_image(img, figsize=(10, 10), gray=False):
    plt.figure(figsize=figsize)

    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def save_image(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # save as bmp,so need bgr format
    cv2.imwrite(path, img)

def rgb_to_yuv(rgb0):
    rgb = rgb0 / 255.0
    y = np.clip(np.dot(rgb, np.array([0.299, 0.587, 0.144])), 0,   1)
    i = np.clip(np.dot(rgb, np.array([0.595716, -0.274453, -0.321263])), -0.5957, 0.5957)
    q = np.clip(np.dot(rgb, np.array([0.211456, -0.522591, 0.311135])), -0.5226, 0.5226)
    yiq = rgb
    yiq[..., 0] = y
    yiq[..., 1] = i
    yiq[..., 2] = q
    return yiq


def yuv_to_rgb(yuv):
    yiq = yuv.copy()
    r = np.dot(yiq, np.array([1.0,  0.956295719758948,  0.621024416465261]))
    g = np.dot(yiq, np.array([1.0, -0.272122099318510, -0.647380596825695]))
    b = np.dot(yiq, np.array([1.0, -1.106989016736491,  1.704614998364648]))
    rgb = yiq
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.clip(rgb, 0.0, 1.0) * 255.0


def random_drop(img, rate=0.8):
    m, n, c = img.shape
    rand = np.random.rand(m, n)
    img_drop = img.copy()
    mask = (rand < rate)  # mask=1, be droped
    mask = mask * np.ones([m, n])
    # for i in range(m):
    #     for j in range(n):
    #         if musk[i,j]==1:
    #             img_drop[i,j,:]=255
    img_drop[mask == 1] = 255

    return img_drop, mask


def loss_func(ans, ground_truth, strategy="MSE"):
    # m, n, c = ans.shape
    # var = (ans - ground_truth) * (ans - ground_truth)
    # MSE = var.sum()
    # MSE = MSE / (m * n * c)
    # if strategy == "MSE":
    #     return MSE
    # elif strategy == "PSNR":
    #     return 10 * np.log10((255 * 255) / MSE)
    # elif strategy == "SSIM":
    #     # ssim = compare_ssim(ans, ground_truth, multichannel=True)
    #     ssim_score=ssim(ans,ground_truth,multichannel=True)
    #     return ssim_score
    MSE=0
    for c in range(3):
        MSE = mse(ground_truth[:,:,c], ans[:,:,c])
        MSE+=MSE
    MSE=MSE/3
    PSNR=10 * np.log10((255 * 255) / MSE)
    ssim_score = ssim(ans, ground_truth, multichannel=True)
    return MSE, PSNR, ssim_score

# img=read_image("dst/debug_output.jpg")
# img=img[27:32,34:40,:]
# show_image(img)
# print(img.reshape(3,5,6))

# img=read_image("dst/lena_output-linear.jpg")
# print(img.shape)
# img=img[445:450,170:175,:]
# show_image(img)
# print(img.reshape(3,5,-1))

# a=[1,2,3,4]

# a=np.ones(5)
# plt.title("aa",fontsize=25)
# a=np.load("a.npy")
# b=2*np.ones(5)
# label=np.array([1,2,3,4,5])
# plt.plot(label,a,label="1")
# plt.plot(label,b,label="2")
# plt.legend(loc=0)
# plt.savefig('aa.jpg')
# plt.show()
# # np.save("a",a)
#
# plt.title("bb")
# label=np.array([1,2,3,4,5])
#
# plt.plot(label,b,label="2")
# plt.legend(loc=0)
# plt.show()
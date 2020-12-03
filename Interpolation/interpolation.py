import sys
import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix, linalg
from utils import *


def rbf_interpolation(img, mask, func="linear", radius=20):
    m, n, _ = img.shape
    step = 2 * radius - 1
    ans = np.zeros(img.shape)
    if func=="gaussian":
        epsilon=0.365         #0.8 0.7 0.5 0.4 0.3 0.35 0.36 0.35
    else:
        epsilon=1

    for c in range(3):
        channel = img[:, :, c].copy()
        print(channel.shape)
        for i in range(0, m, step):
            for j in range(0, n, step):
                num_know = 0  # pixel already known
                x = []
                y = []
                b = []
                for k in range(i - radius, i + radius, 1):
                    for l in range(j - radius, j + radius, 1):
                        if k >= 0 and k < m and l >= 0 and l < n:
                            if mask[k, l] == 0:
                                x.append(k)
                                y.append(l)
                                b.append(channel[k, l])
                                num_know += 1
                if num_know > 1 and num_know < (2 * radius + 1) * (2 * radius + 1):
                    A = np.zeros([num_know, num_know])
                    # w=np.zeros(num_know)
                    for k in range(num_know):
                        for l in range(num_know):
                            r = distance(x[k], y[k], x[l], y[l])
                            A[k, l] = r
                            # A[k,l]=rbf_functions(r,func=func)
                    A = rbf_functions(A, epsilon=epsilon, func=func)
                    w = np.linalg.solve(A, b)
                    # print(w)
                    for k in range(i - radius, i + radius, 1):
                        for l in range(j - radius, j + radius, 1):
                            if k >= 0 and k < m and l >= 0 and l < n:
                                if mask[k, l] == 1:
                                    channel[k, l] = 0
                                    for d in range(num_know):
                                        r = distance(k, l, x[d], y[d])
                                        channel[k, l] += w[d] * rbf_functions(r, epsilon=epsilon ,func=func)
        ans[:, :, c] = channel
    print(ans.shape)
    return ans


def neareat_interpolation(img, mask, radius=4):
    # assert (img.shape==mask.shape)
    ans = np.zeros(img.shape)
    m, n, _ = img.shape

    for c in range(3):
        channel = img[:, :, c].copy()
        for i in range(m):
            for j in range(n):
                N = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1),
                     (i + 1, j + 1)]
                found = 0
                if mask[i, j] == 1:
                    for k, l in N:
                        if 0 <= k < m and 0 <= l < n and mask[k, l] == 0:
                            channel[i, j] = channel[k, l]
                            found = 1
                            break
                    if found == 0:
                        N2 = [(i - 2, j), (i + 2, j), (i, j - 2), (i, j + 2), (i - 2, j + 1), (i - 2, j - 1),
                              (i + 2, j - 1), (i + 2, j + 1), (i - 1, j - 2), (i + 1, j - 2),
                              (i - 1, j + 2), (i + 1, j + 2), (i - 3, j - 3), (i - 3, j + 3), (i + 3, j - 3),
                              (i + 3, j + 3), ]
                        for k, l in N2:
                            if 0 <= k < m and 0 <= l < n and mask[k, l] == 0:
                                channel[i, j] = channel[k, l]
                                found = 1
                                break
                    if found == 0:
                        for k in range(i - radius, i + radius, 1):
                            for l in range(j - radius, j + radius, 1):
                                if (k, l) not in N and (k, l) not in N2:
                                    if 0 <= k < m and 0 <= l < n and mask[k, l] == 0:
                                        channel[i, j] = channel[k, l]
                                        found = 1
                                        break
        ans[:, :, c] = channel
    return ans


def bilinear_interpolation(img, mask):
    m, n, _ = img.shape
    ans = np.zeros(img.shape)
    for c in range(3):
        b = np.zeros(m * n)
        row = []
        col = []
        val = []
        for i in range(m):
            for j in range(n):
                if mask[i, j] == 0:
                    b[i * n + j] = img[i, j, c]
                    row.append(i * n + j)
                    col.append(i * n + j)
                    val.append(1)
                else:
                    N = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1),
                         (i + 1, j + 1)]
                    N1 = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]
                    row.append(i * n + j)
                    col.append(i * n + j)
                    val.append(-1)
                    for k, l in N:
                        if k < 0 or k >= m or l < 0 or l >= n:
                            continue
                        row.append(i * n + j)
                        col.append(k * n + l)
                        if i == 0 or i == m or j == 0 or j == m:
                            val.append(0.2)
                        else:
                            val.append(0.125)
                        # if (k,l) in N1:
                        #     val.append(0.1475)
                        # else:
                        #     val.append(0.1025)
        A = csc_matrix((val, (row, col)))
        LU = linalg.splu(A)
        x = LU.solve(b)
        ans[:, :, c] = x.reshape((m, n))
    ans[mask == 0] = img[mask == 0].copy()
    return ans


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'debug'
    mask_s = 1 if len(sys.argv) > 2 and sys.argv[2] == "mask" else 0
    test_rates = 1 if len(sys.argv) > 3 and sys.argv[3] == "test" else 0
    img = read_image('src/' + name + '.jpg')
    # show_image(img)
    yuv = rgb_to_yuv(img)

    # musk = read_image('src/' + name + '_musk.jpg')
    drop_path = 'dst/' + name + '_drop'
    path_rbf = 'dst/' + name + '_rbf'
    path_near = 'dst/' + name + '_near'
    path_bilin = 'dst/' + name + '_bilin'

    # test_rates=1
    if test_rates == 1:
        rates = np.zeros(9)
        rbf_loss_mse = np.zeros(9)
        near_loss_mse = np.zeros(9)
        bilin_loss_mse = np.zeros(9)
        rbf_loss_psnr = np.zeros(9)
        near_loss_psnr = np.zeros(9)
        bilin_loss_psnr = np.zeros(9)
        rbf_loss_ssim = np.zeros(9)
        near_loss_ssim = np.zeros(9)
        bilin_loss_ssim = np.zeros(9)
        count = 0

        for rate in range(1, 10, 1):
            rate = rate / 10
            img_drop_yuv, mask = random_drop(yuv, rate)

            img_drop = yuv_to_rgb(img_drop_yuv).astype(np.uint8)
            show_image(img_drop)
            appendix1 = "-" + rate.__str__() + ".jpg"
            save_image(img_drop, drop_path + appendix1)

            img_out_yuv_rbf_multiqua = rbf_interpolation(img_drop_yuv, mask, func="multiquadric")
            # img_out_yuv_rbf_tps = rbf_interpolation(img_drop_yuv, mask, func="tps")
            # img_out_yuv_rbf_inverse_quad = rbf_interpolation(img_drop_yuv, mask, func="inverse_multiquad")
            img_out_yuv_near = neareat_interpolation(img_drop_yuv, mask)
            img_out_yuv_bilin = bilinear_interpolation(img_drop_yuv, mask)

            # img_out_rbf_tps = yuv_to_rgb(img_out_yuv_rbf_tps).astype(np.uint8)
            img_out_rbf_multiqua = yuv_to_rgb(img_out_yuv_rbf_multiqua).astype(np.uint8)
            # img_out_rbf_inverse_quad = yuv_to_rgb(img_out_yuv_rbf_inverse_quad).astype(np.uint8)
            img_out_near = yuv_to_rgb(img_out_yuv_near).astype(np.uint8)
            img_out_bilin = yuv_to_rgb(img_out_yuv_bilin).astype(np.uint8)
            show_image(img)
            # show_image(img_out_rbf_tps)
            show_image(img_out_rbf_multiqua)
            # show_image(img_out_rbf_inverse_quad)
            show_image(img_out_near)
            show_image(img_out_bilin)
            # print(img_out)

            appendix = "_" + rate.__str__() + ".jpg"
            # save_image(img_out_rbf_tps, name+"_tps" + appendix)
            save_image(img_out_rbf_multiqua, name + "_multiqua" + appendix)
            # save_image(img_out_rbf_inverse_quad, name + "_inver_quad" + appendix)
            # save_image(img_out_rbf, path_rbf + appendix)
            save_image(img_out_near, path_near + appendix)
            save_image(img_out_bilin, path_bilin + appendix)

            rates[count] = rate
            rbf_loss_mse[count], rbf_loss_psnr[count], rbf_loss_ssim[count] = loss_func(img, img_out_rbf_multiqua)
            # bilin_loss_mse[count], bilin_loss_psnr[count], bilin_loss_ssim[count] = loss_func(img, img_out_rbf_tps)
            # near_loss_mse[count], near_loss_psnr[count], near_loss_ssim[count] = loss_func(img, img_out_rbf_inverse_quad)
            # rbf_loss_mse[count], rbf_loss_psnr[count], rbf_loss_ssim[count] = loss_func(img, img_out_rbf)
            bilin_loss_mse[count], bilin_loss_psnr[count], bilin_loss_ssim[count] = loss_func(img, img_out_bilin)
            near_loss_mse[count], near_loss_psnr[count], near_loss_ssim[count] = loss_func(img, img_out_near)
            count = count + 1

        # np.save("multiqua_loss_mse", rbf_loss_mse)
        # np.save("multiqua_loss_psnr", rbf_loss_psnr)
        # np.save("multiqua_loss_ssim", rbf_loss_ssim)
        # np.save("tps_loss_mse", bilin_loss_mse)
        # np.save("tps_loss_psnr", bilin_loss_psnr)
        # np.save("tps_loss_ssim", bilin_loss_ssim)
        # np.save("inverse_quad_loss_mse", near_loss_mse)
        # np.save("inverse_quad_loss_psnr", near_loss_psnr)
        # np.save("inverse_quad_loss_ssim", near_loss_ssim)
        np.save("rbf_loss_mse", rbf_loss_mse)
        np.save("rbf_loss_psnr", rbf_loss_psnr)
        np.save("rbf_loss_ssim", rbf_loss_ssim)
        np.save("bilin_loss_mse", bilin_loss_mse)
        np.save("bilin_loss_psnr", bilin_loss_psnr)
        np.save("bilin_loss_ssim", bilin_loss_ssim)
        np.save("near_loss_mse", near_loss_mse)
        np.save("near_loss_psnr", near_loss_psnr)
        np.save("near_loss_ssim", near_loss_ssim)

        plt.title("MSE loss", fontsize=25)
        plt.plot(rates, rbf_loss_mse, label="rbf")
        plt.plot(rates, bilin_loss_mse, label="bilin")
        plt.plot(rates, near_loss_mse, label="near")
        # plt.plot(rates, rbf_loss_mse, label="multiquad")
        # plt.plot(rates, bilin_loss_mse, label="tps")
        # plt.plot(rates, near_loss_mse, label="inverse quad")
        plt.legend(loc=0)
        plt.savefig('MSE-loss.jpg')
        plt.show()

        plt.title("PSNR loss", fontsize=25)
        plt.plot(rates, rbf_loss_psnr, label="rbf")
        plt.plot(rates, bilin_loss_psnr, label="bilin")
        plt.plot(rates, near_loss_psnr, label="near")
        # plt.plot(rates, rbf_loss_psnr, label="multiquad")
        # plt.plot(rates, bilin_loss_psnr, label="tps")
        # plt.plot(rates, near_loss_psnr, label="inverse quad")
        plt.legend(loc=0)
        plt.savefig('PSNR-loss.jpg')
        plt.show()

        plt.title("SSIM loss", fontsize=25)
        plt.plot(rates, rbf_loss_ssim, label="rbf")
        plt.plot(rates, bilin_loss_ssim, label="bilin")
        plt.plot(rates, near_loss_ssim, label="near")
        # plt.plot(rates, rbf_loss_ssim, label="multiquad")
        # plt.plot(rates, bilin_loss_ssim, label="tps")
        # plt.plot(rates, near_loss_ssim, label="inverse quad")
        plt.legend(loc=0)
        plt.savefig('SSIM-loss.jpg')
        plt.show()

    # mask_s=1
    if mask_s == 1:
        # mask_ = read_image('src/' + name + '_mask.jpg')
        # mask = (img == mask_).sum(axis=2) != 3
        # img_drop_yuv = rgb_to_yuv(mask_)
        img_drop_yuv, mask = random_drop(yuv, 0.9)
        img_out_yuv_rbf = rbf_interpolation(img_drop_yuv, mask, func="gaussian")
        # img_out_yuv_rbf=yuv.copy()
        img_out_yuv_near = neareat_interpolation(img_drop_yuv, mask)
        img_out_yuv_bilin = bilinear_interpolation(img_drop_yuv, mask)

        img_out_rbf = yuv_to_rgb(img_out_yuv_rbf).astype(np.uint8)
        img_out_near = yuv_to_rgb(img_out_yuv_near).astype(np.uint8)
        img_out_bilin = yuv_to_rgb(img_out_yuv_bilin).astype(np.uint8)
        show_image(img)
        show_image(img_out_rbf)
        show_image(img_out_near)
        show_image(img_out_bilin)

        appendix = ".jpg"
        save_image(img_out_rbf, path_rbf + appendix)
        save_image(img_out_near, path_near + appendix)
        save_image(img_out_bilin, path_bilin + appendix)

        rbf_loss_mse, rbf_loss_psnr, rbf_loss_ssim = loss_func(img, img_out_rbf)
        bilin_loss_mse, bilin_loss_psnr, bilin_loss_ssim = loss_func(img, img_out_bilin)
        near_loss_mse, near_loss_psnr, near_loss_ssim = loss_func(img, img_out_near)
        loss1 = [rbf_loss_mse, bilin_loss_mse, near_loss_mse]
        loss2=[rbf_loss_psnr,bilin_loss_psnr,near_loss_psnr]
        loss3=[rbf_loss_ssim,bilin_loss_ssim,near_loss_ssim]
        print(loss1)
        print(loss2)
        print(loss3)

        # labels=["rbf","bilin","near"]
        # plt.title("MSE loss", fontsize=25)
        # plt.bar(labels, loss1)
        # plt.savefig('MSE-loss.jpg')
        # plt.show()
        #
        # plt.title("PSNR loss", fontsize=25)
        # plt.bar(labels, loss2)
        # plt.savefig('PSNR-loss.jpg')
        # plt.show()
        #
        # plt.title("SSIM loss", fontsize=25)
        # plt.bar(labels, loss3 )
        # plt.savefig('SSIM-loss.jpg')
        # plt.show()


import sys
import numpy as np
from utils import *
from scipy.sparse import csc_matrix, linalg


def colorization_using_optimization(gray, marked):
    assert gray.shape == marked.shape
    assert (gray == marked).sum() > 0

    mask = (gray == marked).sum(axis=2) != 3        #scribbles=1, origin=0 musk->m*n
    gray = rgb_to_yuv(gray)
    marked = rgb_to_yuv(marked)
    m, n, _ = gray.shape

    row = []
    col = []
    val = []
    U0 = np.zeros(m * n)
    V0 = np.zeros(m * n)
    for i in range(m):
        for j in range(n):
            if mask[i, j]:
                U0[i * n + j] = marked[i, j, 1]
                V0[i * n + j] = marked[i, j, 2]
                row.append(i * n + j)
                col.append(i * n + j)
                val.append(1)
            else:
                row.append(i * n + j)
                col.append(i * n + j)
                val.append(1)
                # N->neighborhold
                N = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
                Y = []
                for x, y in N:
                    if x < 0 or x >= m or y < 0 or y >= n:
                        continue
                    Y.append(gray[x, y, 0])
                Y = np.array(Y)
                s = Y.max() - Y.min() + 0.01
                # s=Y.std()+0.01
                W = []
                for x, y in N:
                    if x < 0 or x >= m or y < 0 or y >= n:
                        continue
                    W.append(np.exp(-(gray[i, j, 0] - gray[x, y, 0])**2 / (2 * s**2)))  # weight
                W = np.array(W)
                c = W.sum()
                for x, y in N:
                    if x < 0 or x >= m or y < 0 or y >= n:
                        continue
                    row.append(i * n + j)
                    col.append(x * n + y)
                    w = np.exp(-(gray[i, j, 0] - gray[x, y, 0])**2 / (2 * s**2))
                    val.append(- w / c)             #normalize
                    # val.append(- w)
#    print(len(row), len(col), len(val))
    A = csc_matrix((val, (row, col)))
#    print(m, n, A.shape, A.dtype)

    LU = linalg.splu(A)
    U = LU.solve(U0)
    V = LU.solve(V0)
    # U = np.ones(m * n)
    # V = np.ones(m * n)
  #  print(U.shape, V.shape)

    result = np.zeros((m, n, 3))
    result[:, :, 0] = gray[:, :, 0]
    result[:, :, 1] = U.reshape((m, n))
    result[:, :, 2] = V.reshape((m, n))
 #   print(result.shape)
    result = yuv_to_rgb(result).astype(np.uint8)
    return result


if __name__ == '__main__':

    name = sys.argv[1] if len(sys.argv)>1 else 'baby'
    # name = 'smiley'
    gray = read_image('data/' + name + '.bmp')
    marked = read_image('data/' + name + '_marked.bmp')
    path = 'data/' + name + '_output.bmp'

    result = colorization_using_optimization(gray, marked)
    show_image(result)
    save_image(result, path)

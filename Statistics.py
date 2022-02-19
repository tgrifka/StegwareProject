import pandas as pd
import matplotlib.pyplot as plt
import DCTCalc
import seaborn as sb
import numpy as np

def main():
    dfblocks = pd.DataFrame()
    c = cover_statistics()
    s = steg_statistics()
    fig = sb.kdeplot(c, shade=True, color='b')
    fig = sb.kdeplot(s, shade=True, color='r')
    plt.show()

def toLSB(img):
    LSB = img.copy()
    for x in range(0, 512):
        for y in range(0, 512):
            for z in range(0, 512):
                LSB[x][y][z] = (LSB[x][y][z] % 2)
    return LSB


def count_zeros(list):
    count = 0
    for x in list:
        if x == 0:
            count += 1
    return count


def count_zeros_in_8x8s(img_list):
    count_results = []
    for x1 in range(0, 512, 8):
        for y1 in range(0, 512, 8):
            count = 0
            for i in range(0,3):
                for x2 in range(0,8):
                    for y2 in range(0,8):
                        if img_list[i][x1+x2][y1+y2] == 0:
                            count += 1
                count_results.append(count)
    return count_results

def deviationYCbCr(img, x, y):
    countY = 0
    countCb = 0
    countCr = 0

    for x2 in range(0, 8):
        for y2 in range(0, 8):
            countY += img[0][x + x2][y + y2]
            countCb += img[1][x + x2][y + y2]
            countCr += img[2][x + x2][y + y2]

    maxV = max(countY, countCr, countCb)

    return 3 - (countY/maxV) - (countCb/maxV) - (countCr/maxV)

def differentNeighbors(img, x, y):
    same = 0
    dif = 0
    for x2 in range(x+1, x+7):
        for y2 in range(y+1, y+7):
            res = pixelComparison(img, x2, y2)
            if res == 8:
                same += 1
            else:
                dif += 1

    return same, dif

def pixelComparison(pixelLSB, x, y):
        count = 0
        y = pixelLSB[0][x][y]
        Cb = pixelLSB[1][x][y]
        Cr = pixelLSB[2][x][y]
        # below
        if y == pixelLSB[0][x-1][y-1] and Cb == pixelLSB[1][x-1][y-1] and Cr == pixelLSB[2][x-1][y-1]:
            count += 1
        if y == pixelLSB[0][x][y-1] and Cb == pixelLSB[1][x][y-1] and Cr == pixelLSB[2][x][y-1]:
            count += 1
        if y == pixelLSB[0][x+1][y-1] and Cb == pixelLSB[1][x+1][y-1] and Cr == pixelLSB[2][x+1][y-1]:
            count += 1
        # inline
        if y == pixelLSB[0][x-1][y] and Cb == pixelLSB[1][x-1][y] and Cr == pixelLSB[2][x-1][y]:
            count += 1
        if y == pixelLSB[0][x+1][y] and Cb == pixelLSB[1][x+1][y] and Cr == pixelLSB[2][x+1][y]:
            count += 1
        # above
        if y == pixelLSB[0][x - 1][y + 1] and Cb == pixelLSB[1][x - 1][y + 1] and Cr == pixelLSB[2][x - 1][y + 1]:
            count += 1
        if y == pixelLSB[0][x][y + 1] and Cb == pixelLSB[1][x][y + 1] and Cr == pixelLSB[2][x][y + 1]:
            count += 1
        if y == pixelLSB[0][x + 1][y + 1] and Cb == pixelLSB[1][x + 1][y + 1] and Cr == pixelLSB[2][x + 1][y + 1]:
            count += 1
        return count

def cover_statistics():
    coef, x = DCTCalc.batch_ret_dct_cover(100)
    zero_count = []     # stores 8x8 zero count where 1D is img, second is each blocks count
    for img in coef:
        zero_count.append(count_zeros_in_8x8s(img))
    zero_count_y = []   # stores each 8x8 zero count in a single dimension
    zero_count_x = []   # stores each 8x8's img slot for plotting
    for img in range(0, len(zero_count)):
        for val in zero_count[img]:
            zero_count_y.append(val)
            zero_count_x.append(img)

    fig, ax = plt.subplots()

    vp = ax.violinplot(zero_count, widths=1, showmedians=True)

    plt.xlabel("Image #**")
    plt.ylabel("Number of zeros in an DCT Block")
    plt.title("Cover Image 0s Distribution - Violin Plot")
    plt.figure(figsize=(20, 8), dpi=400)

    plt.show()

    plt.scatter(zero_count_x, zero_count_y)
    plt.xlabel("Image #**")
    plt.ylabel("Number of zeros in an DCT Block")
    plt.title("Cover Image - 0s per 8x8")
    plt.show()
    print("Cover Statistics")
    df_zero_8x8 = pd.DataFrame(zero_count_y)
    print(df_zero_8x8.describe())

    ax = sb.kdeplot(zero_count_y)
    plt.show()
    return zero_count_y



def steg_statistics():
    coef, x = DCTCalc.batch_ret_dct_steg(100)
    zero_count = []
    for img in coef:
        zero_count.append(count_zeros_in_8x8s(img))
    zero_count_y = []
    zero_count_x = []
    for img in range(0, len(zero_count)):
        for val in zero_count[img]:
            zero_count_y.append(val)
            zero_count_x.append(img)

    fig, ax = plt.subplots()

    vp = ax.violinplot(zero_count, widths=1, showmedians=True)

    plt.xlabel("Image #**")
    plt.ylabel("Number of zeros in an DCT Block")
    plt.title("Steg Image 0s Distribution - Violin Plot")
    plt.show()


    plt.scatter(zero_count_x, zero_count_y)
    plt.xlabel("Image #**")
    plt.ylabel("Number of zeros in an DCT Block")
    plt.title("Steg Image - 0s per 8x8")
    plt.show()
    print("Steg Statistics")
    df_zero_8x8 = pd.DataFrame(zero_count_y)
    print(df_zero_8x8.describe())

    sb.kdeplot(zero_count_y)
    plt.show()

    return zero_count_y

if __name__ == "__main__":
    main()

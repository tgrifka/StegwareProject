import math
import numpy as np
from PIL import Image


def main():
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    for i in range(1, 2): #need to do 202
        imageNum = ""
        if i < 10:
            imageNum = "0000"+str(i)
        elif i < 100:
            imageNum = "000" + str(i)
        elif i < 1000:
            imageNum = "00" + str(i)

        try:
            img = Image.open("D:\\Stegware\\alaska2-image-steganalysis\\Cover\\"+imageNum+".jpg")
            img2 = Image.open("D:\\Stegware\\alaska2-image-steganalysis\\JMiPOD\\" + imageNum + ".jpg")
        except:
            continue
        printPathC = "D:\\Stegware\\StegwareProject\\Coefficients\\Cover\\"+imageNum+".txt"
        printPathJMiPOD = "D:\\Stegware\\StegwareProject\\Coefficients\\JMiPOD\\" + imageNum + ".txt"
        printPathcoverjmipod = "D:\\Stegware\\StegwareProject\\Coefficients\\Difference\\cover-jmipod\\" + imageNum + ".txt"
        print(imageNum)

        cover = calcualte_coefficients_from_img(img)
        write_to_file(cover, printPathC)

        steg = calcualte_coefficients_from_img(img2)
        write_to_file(steg, printPathJMiPOD)

        dif = np.subtract(cover, steg)
        write_to_file(dif, printPathcoverjmipod)

    """
    pixelsL = [[29, 255, 29, 76, 76, 76, 76, 76],
               [255, 29, 255, 255, 255, 255, 255, 255],
               [29, 255, 29, 76, 76, 76, 76, 76],
               [255, 255, 255, 255, 255, 255, 255, 255],
               [76, 76, 76, 76, 76, 76, 76, 76],
               [255, 255, 255, 255, 255, 255, 255, 255],
               [76, 76, 76, 76, 76, 76, 76, 76],
               [255, 255, 255, 255, 255, 255, 255, 255]]

    DCTL = DCT(pixelsL, 8, 8)
    print(f"DCTL = {DCTL}")
    pixelsCb = [[256, 0, 256, 85, 85, 85, 85, 85],
                [0, 256, 0, 0, 0, 0, 0, 0],
                [256, 0, 256, 85, 85, 85, 85, 85],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [85, 85, 85, 85, 85, 85, 85, 85],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [85, 85, 85, 85, 85, 85, 85, 85],
                [0, 0, 0, 0, 0, 0, 0, 0]]
    DCTCb = DCT(pixelsCb, 8, 8)
    print(f"DCTCb = {DCTCb}")
    pixelsCr = [[107, 0, 107, 255, 255, 255, 255, 255],
                [0, 107, 0, 0, 0, 0, 0, 0],
                [107, 0, 107, 255, 255, 255, 255, 255],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [255, 255, 255, 255, 255, 255, 255, 255],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [255, 255, 255, 255, 255, 255, 255, 255],
                [0, 0, 0, 0, 0, 0, 0, 0]]
    DCTCr = DCT(pixelsCr, 8, 8)
    print(f"DCTCr = {DCTCr}")

    quantL = QuantizeLuminance(DCTL, 50)
    quantCb = QuantizeChromiance(DCTCr, 50)
    quantCr = QuantizeChromiance(DCTCb, 50)
    """


def check_read():
    x = read_from_file("D:\\Stegware\\StegwareProject\\Coefficients\\Cover\\00001.txt")
    y = batch_calculate_dct(1)[0][0]

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            for k in range(0, len(x[0][0])):
                if x[i][j][k] != y[i][j][k]:
                    print(f"i={i}, j={j}, k={k}, xVal={x[i][j][k]}, yVal={y[i][j][k]}")
                    break
    print("Done")

def read_from_file(path):
    f = open(path, 'r')
    values = []
    set = []
    text = f.read()
    text = text.split('[[')
    for x in text[1].split('['):
        i = []
        for y in x.split():
            if y == '':
                continue
            i.append(int(y.split('.')[0]))
        set.append(i)
    values.append(set)
    set = []
    for x in text[2].split('['):
        i = []
        for y in x.split():
            if y == '':
                continue
            i.append(int(y.split('.')[0]))
        set.append(i)
    values.append(set)
    set = []
    for x in text[3].split('['):
        i = []
        for y in x.split():
            if y == '':
                continue
            i.append(int(y.split('.')[0]))
        set.append(i)
    values.append(set)
    return values


def batch_ret_dct_cover(num_to_ret):
    set = []
    nums = []
    for i in range(1, num_to_ret + 1):
        imageNum = ""
        if i < 10:
            imageNum = "0000" + str(i)
        elif i < 100:
            imageNum = "000" + str(i)
        elif i < 1000:
            imageNum = "00" + str(i)

        printPathC = "D:\\Stegware\\StegwareProject\\Coefficients\\Cover\\" + imageNum + ".txt"

        try:
            cover = read_from_file(printPathC)
            set.append(cover)
            nums.append(i)
        except:
            continue

    return set, nums

def batch_ret_dct_steg(num_to_ret):
    set = []
    nums = []
    for i in range(1, num_to_ret + 1):
        imageNum = ""
        if i < 10:
            imageNum = "0000" + str(i)
        elif i < 100:
            imageNum = "000" + str(i)
        elif i < 1000:
            imageNum = "00" + str(i)

        printPathJMiPOD = "D:\\Stegware\\StegwareProject\\Coefficients\\JMiPOD\\" + imageNum + ".txt"

        try:
            steg = read_from_file(printPathJMiPOD)
            set.append(steg)
            nums.append(i)
        except:
            continue

    return set

def batch_calculate_dct(num_to_calc):
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    set = []
    labels = []
    for i in range(1, num_to_calc+1):
        imageNum = ""
        if i < 10:
            imageNum = "0000"+str(i)
        elif i < 100:
            imageNum = "000" + str(i)
        elif i < 1000:
            imageNum = "00" + str(i)

        printPathC = "D:\\Stegware\\StegwareProject\\Coefficients\\Cover\\" + imageNum + ".txt"
        printPathJMiPOD = "D:\\Stegware\\StegwareProject\\Coefficients\\JMiPOD\\" + imageNum + ".txt"
        printPathcoverjmipod = "D:\\Stegware\\StegwareProject\\Coefficients\\Difference\\cover-jmipod\\" + imageNum + ".txt"

        try:
            print(imageNum)
            try:
                cover = read_from_file(printPathC)
                steg = read_from_file(printPathJMiPOD)
                set.append(cover)
                set.append(steg)
                labels.append("cover")
                labels.append("steg")
            except:
                img = Image.open("D:\\Stegware\\alaska2-image-steganalysis\\Cover\\"+imageNum+".jpg")
                img2 = Image.open("D:\\Stegware\\alaska2-image-steganalysis\\JMiPOD\\" + imageNum + ".jpg")

                cover = calcualte_coefficients_from_img(img)
                write_to_file(cover, printPathC)
                set.append(cover)

                steg = calcualte_coefficients_from_img(img2)
                write_to_file(steg, printPathJMiPOD)
                set.append(steg)

                dif = np.subtract(cover, steg)
                write_to_file(dif, printPathcoverjmipod)

                labels.append("cover")
                labels.append("steg")
        except:
            continue

    return set,labels

def write_to_file(coefficients, path):
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    f = open(path, "w")
    for i in coefficients:
        f.write(str(i))
        f.write("\n")
    f.close()

def return_cofficients(low, high):
    set = []
    labels = []
    for i in range(low, high+1):
        imageNum = ""
        if i < 10:
            imageNum = "0000" + str(i)
        elif i < 100:
            imageNum = "000" + str(i)
        elif i < 1000:
            imageNum = "00" + str(i)

        try:
            img = Image.open("D:\\Stegware\\alaska2-image-steganalysis\\Cover\\" + imageNum + ".jpg")
            img2 = Image.open("D:\\Stegware\\alaska2-image-steganalysis\\JMiPOD\\" + imageNum + ".jpg")
        except:
            continue
        print(imageNum)
        cover = calcualte_coefficients_from_img(img)
        steg = calcualte_coefficients_from_img(img2)
        set.append(cover)
        set.append(steg)
        labels.append("cover")
        labels.append("steg")

    return set,labels

"""
Calculates the DCT2 Coefficients from an input image
Inputs:
    img(Image): The image to calculate the DCT2 Coefficients of, created using Pillow
Returns:
    qLum(2D Matrix): the quantized matrix representing the Quantized DCT2 coefficients for Luminance 
    qCb(2D Matrix): the quantized matrix representing the Quantized DCT2 coefficients for Color Difference Blue
    qCr(2D Matrix): the quantized matrix representing the Quantized DCT2 coefficients for Color Difference Red
"""
def calcualte_coefficients_from_img(img):
    pixels = img.load()
    width, height = img.size
    YCbCr = np.zeros((width, height, 3))
    print("Converting to YCbCr")
    for x in range(0, width):
        for y in range(0, height):
            YCbCrVal = convert_RGB_to_YCBCR(pixels[x, y])
            YCbCr[x][y][0] = YCbCrVal[0]
            YCbCr[x][y][1] = YCbCrVal[1]
            YCbCr[x][y][2] = YCbCrVal[2]
    lum, Cb, Cr = flatten(pixels, width, height)
    shift(lum, width, height)
    qtable = img.quantization
    us0 = un_snake(qtable[0])
    us1 = un_snake(qtable[1])

    #convert YCbCr to DCT Coefficents
    print("Converting YCbCr to DCT2")
    dL = DCT(lum, width, height)
    dCb = DCT(Cb, width, height)
    dCr = DCT(Cr, width, height)

    # Quantizize the values
    print("Quantizing values")
    qLum = quant_using_table(dL, us0, width, height)
    qCb = quant_using_table(dCb, us1, width, height)
    qCr = quant_using_table(dCr, us1, width, height)
    return qLum, qCb, qCr


"""
InPlace - shifts the given array, by -128
Inputs:
    array(2D Matrix): the array to be shifted
    width(int): the width of the matrix
    height(int): the height of the matrix
Returns:
    array(2D Matrix): the original input matrix shifted
"""
def shift(array, width, height):
    for x in range(0, width):
        for y in range(0, height):
            array[x][y] -= 128
    return array

def Coefficient(x):
    if x == 0:
        return 1 / math.sqrt(2)
    else:
        return 1

"""
Converts a pixel from RGB to YCBCr
Inputs:
    pixel(3-tuple): the rgb pixel value
Returns:
    res(3-tuple): the YCbCr representation of the color
"""
def convert_RGB_to_YCBCR(pixel):
    p = [pixel[0], pixel[1], pixel[2]]
    conversion = [[0.299, -0.168935, 0.499813],
                  [0.587, -0.331665, -0.418531],
                  [0.114, 0.50059, -0.081282]]
    res = np.matmul(p, conversion)
    return res

"""
Calculates the DCT values for a large block of pixels
Inputs:
    pixels:
    width:
    height:
Returns:
    dct(2D Matrix): 
"""
def DCT(pixels, width, height):
    dct = np.zeros((width, height))
    for x in range(0, width, 8):
        for y in range(0, height, 8):
            block = DCTBlock(pixels, x, y)
            for i in range(0, 8):
                for j in range(0, 8):
                    dct[x+i, y+j] = block[i][j]
    return dct

def DCTBlock(pixels, startx, starty):
    PI = math.pi
    DCT = np.zeros((8, 8))
    for i in range(0, 8):
        for j in range(0, 8):
            temp = 0.0
            for x in range(0, 8):
                for y in range(0, 8):
                    temp += (math.cos(((2 * x + 1) * i * PI) / (2 * 8)) *
                             math.cos(((2 * y + 1) * j * PI) / (2 * 8)) *
                             pixels[startx + x][starty + y])
            temp *= (1/math.sqrt(2 * 8)) * Coefficient(i) * Coefficient(j)
            DCT[i][j] = round(temp)
    return DCT

"""
Takes a 3D Matrix of Pixels, and returns 3 2D Matrices representing each color channel
Inputs:
    pixels(3D Matrix): The pixel values
    width(int): width of pixels
    height(int): height of pixels
Returns:
    lum(2D Matrix): Matrix representing Luminance Channel
    Cb(2D Matrix): Matrix representing Color Difference Blue Channel
    Cr(2D Matrix): Matrix representing Color Difference Red Channel
"""
def flatten(pixels, width, height):
    lum = np.zeros((width, height))
    Cb = np.zeros((width, height))
    Cr = np.zeros((width, height))
    for x in range(0, width):
        for y in range(0, height):
            lum[x][y] = pixels[x, y][0]
            Cb[x][y] = pixels[x, y][1]
            Cr[x][y] = pixels[x, y][2]

    return lum, Cb, Cr

"""
Quantizize table for Luminance Values
Inputs:
    DCTVals(2D Matrix): The DCT Values for a Luminance color to be quantizized
    QF(int): The Quality Factor to be used
Returns:
    quantized(2D Matrix): Quatizized values for DCTVals
"""
def QuantizeLuminance(DCTVals, QF):
    quant_table = [[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]]

    quantized = np.zeros((8, 8))
    for i in range(0, 8):
        for j in range(0, 8):
            quantized[i][j] = math.floor(DCTVals[i][j] / quant_table[i][j])
    return quantized
"""
Quantizize table for Chromiance Values
Inputs:
    DCTVals(2D Matrix): The DCT Values for a chromiance color to be quantizized
    QF(int): The Quality Factor to be used
Returns:
    quantized(2D Matrix): Quatizized values for DCTVals
"""
def QuantizeChromiance(DCTVals, QF):
    quant_table = [[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]]

    quantized = np.zeros((8, 8))
    for i in range(0, 8):
        for j in range(0, 8):
            quantized[i][j] = math.floor(DCTVals[i][j] / quant_table[i][j])
    return quantized

"""
Quantizes the DCTVals, using a given Quatization table
inputs:
    DCTVals(2D Natrix): The DCTVals to be quatizized
    table(2D Matrix): The Quantization table to be used
    width(int): Width of DCTVals 
    height(int): Height of DCTVals
returns:
    q: Quantizized values from DCTVals using table
"""
def quant_using_table(DCTVals, table, width, height):
    q = np.zeros((width, height))
    for x in range(0, width, 8):
        for y in range(0, height, 8):
            block = quant_using_table_block(DCTVals, table, x, y)
            for i in range(0, 8):
                for j in range(0, 8):
                    q[x+i, y+j] = block[i][j]
    return q

"""
Quantizes a single 8x8 block
inputs:
    DCTVals(2D Natrix): The DCTVals to be quatizized
    table(2D Matrix): The Quantization table to be used
    startx(int): Starting X Value
    starty(int): Starting Y value
returns:
    quantized: The block that is quantized
"""
def quant_using_table_block(DCTVals, table, startx, starty):
    quantized = np.zeros((8, 8))
    for i in range(0, 8):
        for j in range(0, 8):
            quantized[i][j] = math.floor(DCTVals[startx + i][starty + j] / table[i][j])
    return quantized

"""
UnSakes the given quant list
Inputs:
    l(list): list of quantization values to be unsnaked
Returns:
    u(2D Matrix): the same list in the form of a 2D matrix, unsnaked
"""
def un_snake(l):
    u = np.zeros((8, 8))
    u[0, 0] = l[0]
    u[0, 1] = l[1]
    u[1, 0] = l[2]
    u[2, 0] = l[3]
    u[1, 1] = l[4]
    u[0, 2] = l[5]
    u[0, 3] = l[6]
    u[1, 2] = l[7]
    u[2, 1] = l[8]
    u[3, 0] = l[9]
    u[4, 0] = l[10]
    u[3, 1] = l[11]
    u[2, 2] = l[12]
    u[1, 3] = l[13]
    u[0, 4] = l[14]
    u[0, 5] = l[15]
    u[1, 4] = l[16]
    u[2, 3] = l[17]
    u[3, 2] = l[18]
    u[4, 1] = l[19]
    u[5, 0] = l[20]
    u[6, 0] = l[21]
    u[5, 1] = l[22]
    u[4, 2] = l[23]
    u[3, 3] = l[24]
    u[2, 4] = l[25]
    u[1, 5] = l[26]
    u[0, 6] = l[27]
    u[0, 7] = l[28]
    u[1, 6] = l[29]
    u[2, 5] = l[30]
    u[3, 4] = l[31]
    u[4, 3] = l[32]
    u[5, 2] = l[33]
    u[6, 1] = l[34]
    u[7, 0] = l[35]
    u[7, 1] = l[36]
    u[6, 2] = l[37]
    u[5, 3] = l[38]
    u[4, 4] = l[39]
    u[3, 5] = l[40]
    u[2, 6] = l[41]
    u[1, 7] = l[42]
    u[2, 7] = l[43]
    u[3, 6] = l[44]
    u[4, 5] = l[45]
    u[5, 4] = l[46]
    u[6, 3] = l[47]
    u[7, 2] = l[48]
    u[7, 3] = l[49]
    u[6, 4] = l[50]
    u[5, 5] = l[51]
    u[4, 6] = l[52]
    u[3, 7] = l[53]
    u[4, 7] = l[54]
    u[5, 6] = l[55]
    u[6, 5] = l[56]
    u[7, 4] = l[57]
    u[7, 5] = l[58]
    u[6, 6] = l[59]
    u[5, 7] = l[60]
    u[6, 7] = l[61]
    u[7, 6] = l[62]
    u[7, 7] = l[63]
    return u

def three_to_one(z):
    l = []
    for i in range(0, 3):
        for x in range(0, 512):
            for y in range(0, 512):
                l.append(z[i][x][y])
    return l

def flatten_all(x):
    for i in range(0, len(x)):
        x[i] = three_to_one(x[i])
    return x

if __name__ == '__main__':
    main()

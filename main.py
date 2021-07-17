from blind_watermark import WaterMark
import glob


if __name__ == '__main__':
    bw = WaterMark(password_wm=1, password_img=1)
    bw.read_img('D:/Stegware/alaska2-image-steganalysis/Cover/00111.jpg')
    wm = 'Embeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded ' \
         'MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded ' \
         'MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded ' \
         'MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded MessageEmbeded '
    bw.read_wm(wm, mode='str')
    bw.embed('D:/Stegware/alaska2-image-steganalysis/WaterMarked/00111WM.png')
    files = glob.glob('D:/Stegware/alaska2-image-steganalysis/Cover/*.jpg')
    """
    for i in range(0, 100):
        bw.read_img(files[i])
        fName = files[i][0:39]+'WaterMarked/'+files[i][45:50]+'WM.png'
        print(fName)
        bw.embed(fName)
    """



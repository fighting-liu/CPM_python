import numpy as np
from cStringIO import StringIO
import PIL.Image
from IPython.display import Image, display
import cv2 as cv

def showBGRimage(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
#     a[:,:,[0,2]] = a[:,:,[2,0]] # for B,G,R order
#     f = StringIO()
#     PIL.Image.fromarray(a).save(f, fmt)
#     display(Image(data=f.getvalue()))
    cv.imshow('test win', a)
    cv.waitKey(0) 

def showmap(a, fmt='png'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)): 
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
    return c

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out

def padRightDownCorner(img):
    """
    As the network is downsizing the input image by 8x(three maxpools), we just need to pad the image on right and bottom to make its width 
    and height be multiplies of 8. As we only pad values in right and bottom, it returns body coordinates the same without padding.
    """
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%8==0) else 8 - (h % 8) # down
    pad[3] = 0 if (w*8==0) else 8 - (w % 8) # right
    
    npad = ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0))
    img_padded = np.pad(
        img, pad_width=npad, mode='constant', constant_values=128)    

    return img_padded


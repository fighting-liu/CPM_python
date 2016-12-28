import cv2 as cv 
import numpy as np
import scipy
import time
import util
import copy
import math

from config import cfg
from net_reader import *
# from demo import canvas

"""
Two-phase algorithm: 
(1)person detection in an image and return body center for each person
(2)multiple-person pose estimations 
"""

def showSkeleton(imageToTest, prediction, num_people):
    """Show skeleton of the test image and draw it.
    
    Argus:
        imageToTest: Image on which to draw the skeleton.(ps: This is scaled image during body detection.)
        prediction: Coordinates prediction on imageToTest.
        num_people: Number of persons in the image.        
    """
    limbs = cfg['limbs']
    stickwidth = 6
    colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
    [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]] # note BGR ...
    canvas = imageToTest.copy()
    for p in range(num_people):
        for part in range(cfg['np']):
            cv.circle(canvas, (int(prediction[part, 1, p]), int(prediction[part, 0, p])), 3, (0, 0, 0), -1)
        for l in range(limbs.shape[0]):
            cur_canvas = canvas.copy()
            X = prediction[limbs[l,:]-1, 0, p]
            Y = prediction[limbs[l,:]-1, 1, p]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[l])
            canvas = canvas * 0.4 + cur_canvas * 0.6 
    util.showBGRimage(canvas)    
    cv.imwrite('/home/test/tangyudi/liuhuawei/showtime/JointsDetection/skelelon/4.jpg', canvas)
    
def produceCenterLabelMap(boxsize, sigma):
    """    Computing the heatmap channel.

    Args:
        im_size: Image size, which is also boxsize.
        x: X coordinate of center of box. 
        y: Y coordinate of center of box. 
        sigma: Variance of gaussian function.

    Returns:
        label: Heatmap channel.   
    """
    test_X = np.linspace(1, boxsize, boxsize)
    test_Y = np.linspace(1, boxsize, boxsize)
    [X, Y] = np.meshgrid(test_X, test_Y)
    X = X - boxsize/2
    Y = Y - boxsize/2
    D2 = np.square(X) + np.square(Y)
    Exponent = D2 / (2.0 * sigma * sigma)
    label = np.exp(-Exponent)
    return label

def cropImg(x, y, num_people, boxsize, imageToTest):
    '''Cropping regions of interest on the image and padding with 128(mean value) for out-of-boundary region.
    
    Argus:
        x, y: coordinates of body center
        num_people: Number of persons in the image.
        boxsize: Box size.
        imageToTest: Scaled image during body detection.
    Returns:
        person_image: Boxes fed to CPM.(boxsize * boxsize * 3 * num_people) 
    '''
    person_image = np.ones((boxsize, boxsize, 3, num_people)) * 128
    h, w, _ = imageToTest.shape 
    for p in xrange(num_people):
        x_left = np.max((x[p] - boxsize/2, 0))
        x_right = np.min((x[p] + boxsize/2, w-1))
        y_upper = np.max((y[p] - boxsize/2, 0))
        y_down = np.min((y[p] + boxsize/2, h-1))

        person_image[boxsize/2-(y[p]-y_upper):boxsize/2+(y_down-y[p]), boxsize/2-(x[p]-x_left):boxsize/2+(x_right-x[p]), :, p] = \
            imageToTest[y_upper:y_down, x_left:x_right, :]           
    return person_image

def cpm_multi(boxsize, person_image, gaussian_map, pose_net):
    """Main function to compute confidence maps in batch mode.
    
    Argus:
        boxsize: w and h of input image.
        person_image: image data of shape (boxsize, boxsize, 3, N).
        gaussian_map: gaussian value of shape (boxsize, boxsize).
        pose_net: caffe object for pose detection.
    Returns:
        output_blobs_array: score map of shape (N, 15, 46, 46).
    """
    H, W, C, N = person_image.shape
    output_blobs_array = [dict() for dummy in range(N)]
    input_data = np.zeros((H, W, C+1, N))
    ##Squeeze data range to [-0.5, 0.5]
    input_data[:, :, 0:3, :] = person_image/256.0 - 0.5
    ##Add gaussian maps
    input_data[:, :, 3, :] = gaussian_map[:, :, np.newaxis]
    pose_net.blobs['data'].reshape(*(N, C+1, H, W))
    pose_net.blobs['data'].data[...] = np.transpose(np.float32(input_data), (3,2,0,1))
    ##output_blobs_array[p] of shape (N, 15, 46, 46)
    output = copy.deepcopy(pose_net.forward()['Mconv7_stage6'])
    for p in xrange(N):
        output_blobs_array[p] = output[np.newaxis, p, :, :]
    return output_blobs_array 

def cpm_que(boxsize, person_image, gaussian_map, pose_net):
    """Main function to compute confidence maps in queue mode.
    
    Argus:
        boxsize: w and h of input image.
        person_image: image data of shape (boxsize, boxsize, 3, N).
        gaussian_map: gaussian value of shape (boxsize, boxsize).
        pose_net: caffe object for pose detection.
    Returns:
        output_blobs_array: score map of shape (N, 15, 46, 46).
    """    
    _, _, _, N = person_image.shape
    output_blobs_array = [dict() for dummy in range(N)]
    for p in range(N):
        input_4ch = np.ones((boxsize, boxsize, 4))
        ##Squeeze data range to [-0.5, 0.5]        
        input_4ch[:,:,0:3] = person_image[:,:,:,p]/256.0 - 0.5 
        input_4ch[:,:,3] = gaussian_map
        pose_net.blobs['data'].data[...] = np.transpose(np.float32(input_4ch[:,:,:,np.newaxis]), (3,2,0,1))
        ##Output_blobs_array[p] of shape (1, 15, 46, 46)
        output_blobs_array[p] = copy.deepcopy(pose_net.forward()['Mconv7_stage6']) 
    return output_blobs_array

def bodyDetect(oriImg, boxsize, detConf, person_net): 
    """A body detector to detect body center
    Args:
        oriImg: The image data on which to detect body center.
        boxsize: Scale height of image to boxsize.
        detConf: Above which we think it will be a person, a hyper-parameter.
        person_net: Caffe model object of the body detection net.
        
    Returns:
        imageToTest: Scaled image.
        scale: imageToTest = oriImg * scale.
        x, y: coordinates of body center
    """  
    ##Scale height to match the boxsize
    scale = boxsize/(oriImg.shape[0] * 1.0)
    imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    
    ##Pad image to fit detection input, that can be divided by 8
    imageToTest_padded = util.padRightDownCorner(imageToTest)
    
    person_net.blobs['image'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))    
    ##Squeeze data range to [-0.5, 0.5]
    person_net.blobs['image'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5    
    output_blobs = person_net.forward()
    ##Remove single-dimensional entry 
    person_map = np.squeeze(person_net.blobs[output_blobs.keys()[0]].data)
        
    ##Location with high confidence with be returned, fx=8, because only 3 max pools in deploy file 
    person_map_resized = cv.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    data_max = scipy.ndimage.filters.maximum_filter(person_map_resized, size=3)
    maxima = (person_map_resized == data_max)
    diff = (data_max > detConf)
    maxima[diff == 0] = 0
    x = np.nonzero(maxima)[1]
    y = np.nonzero(maxima)[0]
    
    return imageToTest, scale, x, y    

def getTimeByStamp(beg, end, mode):
    t = end - beg
    if 'SEC' == mode.upper():
        return t*1.0
    elif 'MIN' == mode.upper():
        return t/60.0
    elif 'HOUR' == mode.upper():
        return t/3600.0
    return t  

def real_two(person_net, pose_net):
    video_capture = cv.VideoCapture(0)
    while True:
        _, frame = video_capture.read()    
        begTime = time.time()
        pred = applyModel(frame, person_net, pose_net, show_skeleton=False)
        canvas = frame.copy()
        if pred is not None:
            num_people = pred.shape[2]  
            limbs = cfg['limbs']
            stickwidth = 6
            colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0], \
                        [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]] # note BGR ...                
            for p in range(num_people):
                for part in range(cfg['np']):
                    cv.circle(canvas, (int(pred[part, 1, p]), int(pred[part, 0, p])), 3, (0, 0, 0), -1)
                    for l in range(limbs.shape[0]):
                        cur_canvas = canvas.copy()
                        X = pred[limbs[l,:]-1, 0, p]
                        Y = pred[limbs[l,:]-1, 1, p]
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                        polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                        cv.fillConvexPoly(cur_canvas, polygon, colors[l])
                        canvas = canvas * 0.4 + cur_canvas * 0.6 
            canvas = np.uint8(np.clip(canvas, 0, 255))    
#             for p in range(num_people):
#                 for x, y in pred[:, :, p]:
#                     cv.circle(frame, (int(y), int(x)), 5,(0, 0, 225), -1)        
        t = getTimeByStamp(begTime, 
                           time.time(), 'SEC')
        print("\tJoints detection time: %f sec"%(t))
        cv.imshow('Video', canvas)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break      
    
def real_time_demo(person_net, pose_net):
    video_capture = cv.VideoCapture(0)
    while True:
        _, frame = video_capture.read()    
        begTime = time.time()
        pred = applyModel(frame, person_net, pose_net, show_skeleton=False)
        if pred is not None:
            num_people = pred.shape[2]  
            for p in range(num_people):
                for x, y in pred[:, :, p]:
                    cv.circle(frame, (int(y), int(x)), 5,(0, 0, 225), -1)        
        t = getTimeByStamp(begTime, 
                           time.time(), 'SEC')
        print("\tJoints detection time: %f sec"%(t))
        cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break        
            
def applyModel(oriImg, person_net, pose_net, show_skeleton=False):       
    boxsize = cfg['boxsize']
    npart = cfg['np']
    sigma = cfg['sigma']
    detConf = cfg['detectionConfidence']
    
    ##Change it, if we use other person detector    
    imageToTest, scale, x, y = bodyDetect(oriImg, boxsize, detConf, person_net)    
    num_people = x.size   
    ##No body centers detected
    if num_people == 0:
        return None
    
    ##Make cpm-input for each body 
    person_image = cropImg(x, y, num_people, boxsize, imageToTest)           
    gaussian_map = produceCenterLabelMap(boxsize, sigma)   
    ##Out of GPU memory danger for batch_size more than 32
    if num_people <= 32:    
        output_blobs_array = cpm_multi(boxsize, person_image, gaussian_map, pose_net)
    else:
        output_blobs_array = cpm_que(boxsize, person_image, gaussian_map, pose_net)   

    ##Compute coordinates with large confidence for each part in each person   
    prediction = np.zeros((npart, 2, num_people))    
    for p in range(num_people):
        for part in range(npart):
            part_map = output_blobs_array[p][0, part, :, :]
            part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
            prediction[part,:,p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
        prediction[:,0,p] = prediction[:,0,p] - (boxsize/2) + y[p]
        prediction[:,1,p] = prediction[:,1,p] - (boxsize/2) + x[p] 
    ##Whether to show skeleton
    if show_skeleton:
        showSkeleton(imageToTest, prediction, num_people)          
    ##Turn back to coordinates in original image    
    prediction = prediction / scale   

    return  prediction
        
if __name__ == '__main__':
    person_net, pose_net = load_net() 
    real_time_demo(person_net, pose_net)    

# #     real_two(person_net, pose_net)
# 
#     img_path = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/sample_image/7.jpg'     
#     start_time = time.time()  
#     img_data = cv.imread(img_path)    
#    
#     pred = applyModel(oriImg=img_data, person_net=person_net, pose_net=pose_net, show_skeleton=True)
#     print('Gross time:  %.2f ms.' % (1000 * (time.time() - start_time)))
#      
#     if pred is not None: 
#         num_people = pred.shape[2]   
#         for p in range(num_people):
#             for x, y in pred[:, :, p]:
#                 cv.circle(img_data, (int(y), int(x)), 5,(0, 0, 225), -1)
#         win = cv.namedWindow('test win', flags=0)
#     cv.imshow('test win', img_data)
# #     cv.imwrite('/home/test/tangyudi/liuhuawei/showtime/JointsDetection/skelelon/3.jpg', img_data)
#     cv.waitKey(0)        
#         
        
        
        
          
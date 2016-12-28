import cv2 as cv 
import numpy as np
import scipy
import time
import util
import copy
import math

from config import cfg
from net_reader import *
from numpy import dtype
# from demo import canvas

"""
Two-phase algorithm: 
(1)person detection in an image and return body center for each person
(2)multiple-person pose estimations 
"""  
    
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
    confidence = np.zeros((npart, 1, num_people)) 
    final_pred = np.zeros((npart, 2, num_people))   
    for p in range(num_people):
        for part in range(npart):
            part_map = output_blobs_array[p][0, part, :, :]
            part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
            prediction[part, :, p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
            confidence[part, :, p] = part_map_resized.max()
        ## exchange x and y coordinates to meet following requirements    
        final_pred[:, 0, p] = prediction[:, 1, p] - (boxsize/2) + x[p]
        final_pred[:, 1, p] = prediction[:, 0, p] - (boxsize/2) + y[p]                  
    return  (imageToTest, final_pred, confidence,)

def add_border(logo, target_H, target_W):
    ##make sure logo height is less than 368.
    h, w = logo.shape[:2]
    new_logo = np.zeros((target_H, target_W, 4))
    if h <= target_H and w <= target_W:
        new_logo[:h, :w, :] = logo
        scale = 1
    elif w > target_W:
        logo = cv.resize(logo, (int(target_W), int(h*(target_W*1.0 / w)))) 
        new_logo[:logo.shape[0], :logo.shape[1], :] = logo 
        scale = target_W * 1.0 / w
    else:
        print "Unexpected error in add_border!!"             
    return new_logo, scale

def isValid(pred, conf, idx):
    ##1.People should face the camera
    if pred[cfg.PART_TO_INDEX['Rsho'], 0, 0] > pred[cfg.PART_TO_INDEX['Lsho'], 0, 0]:
        print "Please turn around!!!"
        return False
    
    ##2.Necessary joints are apparent
    #shaonv dressing
    if idx == 1:
        joints_index = cfg.CHARTLET.SN_JOINTSINDEX 
        joints_conf = conf[joints_index, :, 0]
        if not np.alltrue(joints_conf > cfg.JOINTS_CONFIDENCE):
            print "You should show your HEAD,NECK,RSHO,LSHO,RHIP,LHIP out for shaonv dressing!!!"
            return False
    #falao dressing    
    if idx == 2:
        joints_index = cfg.CHARTLET.FL_JOINTSINDEX 
        joints_conf = conf[joints_index, :, 0]
        if not np.alltrue(joints_conf > cfg.JOINTS_CONFIDENCE):
            print "You should show your HEAD,NECK,RSHO,LSHO,RHIP,RKNE,LHIP out for falao dressing!!!"
            return False     
    #caoqun dressing    
    if idx == 3:
        joints_index = cfg.CHARTLET.CQ_JOINTSINDEX 
        joints_conf = conf[joints_index, :, 0]
        if not np.alltrue(joints_conf > cfg.JOINTS_CONFIDENCE):
            print "You should show your RSHO,LSHO,RHIP,RKNE,LHIP out for caoqun dressing!!!"
            return False                  
    return True
     
def create_mask(mat):
    base_mask = mat[:, :, 3] > cfg.CHARTLET.TRA_THRE
    mask = np.zeros((base_mask.shape[0], base_mask.shape[1], 3), dtype=np.bool)
    for i in xrange(3):
        mask[:, :, i] = base_mask
    return mask    

def clip_border(img_data, start_H, start_W, end_H, end_W):
    H, W = img_data.shape[:2]
    start_H = np.maximum(start_H, 0)
    start_W = np.maximum(start_W, 0)
    end_H = np.minimum(end_H, H)
    end_W = np.minimum(end_W, W)
    return start_H, start_W, end_H, end_W
        
def tietu(img_data, pred, idx): 
    H, W, _ = img_data.shape                    
    ####shaonv zhuang          
    if idx == 1:
        logo = cv.imread(cfg.CHARTLET.SN_CLOTH_PATH, -1)              
        logo, scale = add_border(logo, H, W)
        pts1 = np.array([cfg.CHARTLET.SN_CLOTH_RSHO, cfg.CHARTLET.SN_CLOTH_LSHO,\
                          cfg.CHARTLET.SN_CLOTH_RHIP], dtype=np.float32) * scale                     
        pts2 = np.array([pred[cfg.PART_TO_INDEX['Rsho'], :, 0], pred[cfg.PART_TO_INDEX['Lsho'], :, 0],\
                         pred[cfg.PART_TO_INDEX['Rhip'], :, 0]], dtype=np.float32)
        M = cv.getAffineTransform(pts1, pts2)
        res = cv.warpAffine(logo, M, (logo.shape[1], logo.shape[0]))
        msk_ = create_mask(res)     
        np.copyto(img_data, res[:, :, :3].astype(np.uint8), where=msk_)

        logo_2 = cv.imread(cfg.CHARTLET.SN_NECK_PATH, -1)
        sho_to_sho_w = np.abs(pred[cfg.PART_TO_INDEX['Lsho'], 0, 0] - pred[cfg.PART_TO_INDEX['Rsho'], 0, 0])
        head_to_neck_h = np.abs(pred[cfg.PART_TO_INDEX['head'], 1, 0] - pred[cfg.PART_TO_INDEX['neck'], 1, 0])
        logo_2 = cv.resize(logo_2, (int(sho_to_sho_w*0.5), int(head_to_neck_h*0.4)))
        neck = pred[cfg.PART_TO_INDEX['neck'], :, 0]        
        h_l, w_l = logo_2.shape[:2]
        start_H = neck[1] 
        start_W = neck[0] - w_l/2         
        end_H = start_H + h_l
        end_W = start_W + w_l    
        img_roi = img_data[start_H:end_H, start_W:end_W] 
        msk_ = create_mask(logo_2)
        np.copyto(img_roi, logo_2[:, :, :3], where=msk_)
    
        logo_3 = cv.imread(cfg.CHARTLET.SN_BRAID_PATH, -1)
        sho_to_sho_w = np.abs(pred[cfg.PART_TO_INDEX['Lsho'], 0, 0] - pred[cfg.PART_TO_INDEX['Rsho'], 0, 0])    
        head_to_hip_h = np.abs(pred[cfg.PART_TO_INDEX['head'], 1, 0] - pred[cfg.PART_TO_INDEX['Rhip'], 1, 0])    
        head_to_neck_h = np.abs(pred[cfg.PART_TO_INDEX['head'], 1, 0] - pred[cfg.PART_TO_INDEX['neck'], 1, 0])        
        logo_3 = cv.resize(logo_3, (int(sho_to_sho_w*2), int(head_to_hip_h)))
        head = pred[cfg.PART_TO_INDEX['head'], :, 0]
        h_l, w_l = logo_3.shape[:2]
        start_H = head[1]-int(head_to_neck_h/3)
        start_W = head[0] - w_l/2
        end_H = start_H + h_l
        end_W = start_W + w_l
        ##clip the crossing border 
        start_H_n, start_W_n, end_H_n, end_W_n = clip_border(img_data, start_H, start_W, end_H, end_W)       
        logo_3 = logo_3[(start_H_n-start_H):(start_H_n-start_H)+(end_H_n-start_H_n), \
                           (start_W_n-start_W):(start_W_n-start_W)+(end_W_n-start_W_n)]               
        img_roi = img_data[start_H_n:end_H_n, start_W_n:end_W_n]       
        msk_ = create_mask(logo_3)
        np.copyto(img_roi, logo_3[:, :, :3], where=msk_) 
         

    ####falao dressing        
    if idx == 2:
        logo = cv.imread(cfg.CHARTLET.FL_DRESS_PATH, -1)            
        logo, scale = add_border(logo, H, W)
        pts1 = np.array([cfg.CHARTLET.FL_DRESS_RHIP, cfg.CHARTLET.FL_DRESS_LHIP, \
                         cfg.CHARTLET.FL_DRESS_RKNE], dtype=np.float32) * scale            
        pts2 = np.array([pred[cfg.PART_TO_INDEX['Rhip'], :, 0], pred[cfg.PART_TO_INDEX['Lhip'], :, 0], \
                         pred[cfg.PART_TO_INDEX['Rkne'], :, 0]], dtype=np.float32)
        M = cv.getAffineTransform(pts1, pts2)
        res = cv.warpAffine(logo, M, (logo.shape[1], logo.shape[0]))      
        msk_ = create_mask(res)     
        np.copyto(img_data, res[:, :, :3].astype(np.uint8), where=msk_)    
        
        logo_2 = cv.imread(cfg.CHARTLET.FL_CAP_PATH, -1)        
        logo_2, scale = add_border(logo_2, H, W)      
        pts1 = np.array([cfg.CHARTLET.FL_CAP_HEAD, cfg.CHARTLET.FL_CAP_RSHO, \
                         cfg.CHARTLET.FL_CAP_LSHO], dtype=np.float32) * scale                
        pts2 = np.array([pred[cfg.PART_TO_INDEX['head'], :, 0], pred[cfg.PART_TO_INDEX['Rsho'], :, 0], \
                         pred[cfg.PART_TO_INDEX['Lsho'], :, 0]], dtype=np.float32)
        M = cv.getAffineTransform(pts1, pts2)
        res = cv.warpAffine(logo_2, M, (logo_2.shape[1], logo_2.shape[0]))                
        msk_ = create_mask(res)
        np.copyto(img_data, res[:, :, :3].astype(np.uint8), where=msk_)
        
        logo_3 = cv.imread(cfg.CHARTLET.FL_COLLAR_PATH, -1)
        sho_to_sho_w = np.abs(pred[cfg.PART_TO_INDEX['Lsho'], 0, 0] - pred[cfg.PART_TO_INDEX['Rsho'], 0, 0])
        head_to_neck_h = np.abs(pred[cfg.PART_TO_INDEX['head'], 1, 0] - pred[cfg.PART_TO_INDEX['neck'], 1, 0])
        logo_3 = cv.resize(logo_3, (int(sho_to_sho_w), int(head_to_neck_h)))
        neck = pred[cfg.PART_TO_INDEX['neck'], :, 0]   
        h_l, w_l = logo_3.shape[:2]
        start_H = neck[1] 
        start_W = neck[0] - w_l/2 
        end_H = start_H + h_l
        end_W = start_W + w_l         
        img_roi = img_data[start_H:end_H, start_W:end_W]       
        msk_ = create_mask(logo_3)
        np.copyto(img_roi, logo_3[:, :, :3], where=msk_)
        
    ####caoqun zhuang   
    if idx == 3:
        logo = cv.imread(cfg.CHARTLET.CQ_DRESS_PATH, -1)                     
        logo, scale = add_border(logo, H, W)      
        pts1 = np.array([cfg.CHARTLET.CQ_DRESS_RHIP, cfg.CHARTLET.CQ_DRESS_LHIP,\
                          cfg.CHARTLET.CQ_DRESS_RKNE], dtype=np.float32) * scale 
        pts2 = np.array([pred[cfg.PART_TO_INDEX['Rhip'], :, 0], pred[cfg.PART_TO_INDEX['Lhip'], :, 0], \
                         pred[cfg.PART_TO_INDEX['Rkne'], :, 0]], dtype=np.float32)
        M = cv.getAffineTransform(pts1, pts2)        
        res = cv.warpAffine(logo, M, (logo.shape[1], logo.shape[0]))            
        msk_ = create_mask(res)
        np.copyto(img_data, res[:, :, :3].astype(np.uint8), where=msk_)     
               
        logo_2 = cv.imread(cfg.CHARTLET.CQ_BRA_PATH, -1) 
        sho_to_sho_w = np.abs(pred[cfg.PART_TO_INDEX['Lsho'], 0, 0] - pred[cfg.PART_TO_INDEX['Rsho'], 0, 0])
        sho_to_hip_h = np.abs(pred[cfg.PART_TO_INDEX['Rhip'], 1, 0] - pred[cfg.PART_TO_INDEX['Rsho'], 1, 0]) / 1.5
        logo_2 = cv.resize(logo_2, (int(sho_to_sho_w), int(sho_to_hip_h)))
        h_l, w_l = logo_2.shape[:2]       
        ##in case, people are not faced to photo.
        start_H = np.minimum(pred[cfg.PART_TO_INDEX['Rsho'], 1, 0], pred[cfg.PART_TO_INDEX['Lsho'], 1, 0])  
        start_W = np.minimum(pred[cfg.PART_TO_INDEX['Rsho'], 0, 0], pred[cfg.PART_TO_INDEX['Lsho'], 0, 0])
        end_H = start_H + h_l
        end_W = start_W + w_l          
        img_roi = img_data[start_H:end_H, start_W:end_W]       
        msk_ = create_mask(logo_2)
        np.copyto(img_roi, logo_2[:, :, :3].astype(np.uint8), where=msk_)         
        
#     if pred is not None: 
#         num_people = pred.shape[2]   
#         for p in range(num_people):
#             for x, y in pred[:, :, p]:
#                 cv.circle(img_data, (int(x), int(y)), 5,(0, 0, 225), -1)                
    cv.imwrite('./1_1.png', img_data)
    cv.imshow('1', img_data)
    cv.waitKey(0)     
                     
if __name__ == '__main__':
    person_net, pose_net = load_net() 
    img_path = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/sample_image/10.jpg'     
    start_time = time.time()  
    img_data = cv.imread(img_path)     
    res = applyModel(oriImg=img_data, person_net=person_net, pose_net=pose_net, show_skeleton=False)
    if res is None:
        print 'Nothing found in image!!'
    else:
        imageToTest, pred, conf = res
        idx = 1
        if isValid(pred, conf, idx):
            print('Gross time:  %.2f ms.' % (1000 * (time.time() - start_time)))
            tietu(imageToTest, pred, idx)      
from easydict import EasyDict as edict
import numpy as np

__C = edict()
# get config by:
# from config import cfg
cfg = __C

__C.boxsize = 368 #boxsize of cpm input
__C.np = 14 #number of parts to detect
__C.sigma = 21 #gaussian map std
__C.detectionConfidence = 0.5 #confidence threshold for body detection

__C.use_gpu = True
__C.GPUdeviceNumber = 0
   
__C.caffemodel_person = '../../model/_trained_person_MPI/pose_iter_70000.caffemodel'
__C.deployFile_person = '../../model/_trained_person_MPI/pose_deploy_copy_4sg_resize.prototxt'
__C.caffemodel_cpm = '../../model/_trained_MPI/pose_iter_320000.caffemodel'
__C.deployFile_cpm = '../../model/_trained_MPI/pose_deploy_resize.prototxt'
__C.description = 'MPII 3 stage 2 level'

__C.limbs = np.array([1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14]).reshape((9, 2))
__C.part_str = ['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'bkg']

#####chartlet part
__C.PART_TO_INDEX = {b:a for a, b in enumerate(__C.part_str[:-1])}
__C.CHARTLET = edict()
__C.CHARTLET.TRA_THRE = 0.9
__C.JOINTS_CONFIDENCE = 0.3
##SHAONV PATH
__C.CHARTLET.SN_CLOTH_PATH = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/python/tietu/shaonv/cloth_1.png'
__C.CHARTLET.SN_NECK_PATH = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/python/tietu/shaonv/neck.png'
__C.CHARTLET.SN_BRAID_PATH = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/python/tietu/shaonv/braid2_1.png'
__C.CHARTLET.SN_CLOTH_RSHO = [30, 16] 
__C.CHARTLET.SN_CLOTH_LSHO = [184, 16] 
__C.CHARTLET.SN_CLOTH_RHIP = [68, 170]
__C.CHARTLET.SN_BRAID_HEAD = [114, 30] 
__C.CHARTLET.SN_BRAID_RHIP = [48, 228] 
__C.CHARTLET.SN_BRAID_LHIP = [190, 228]
__C.CHARTLET.SN_JOINTSINDEX = np.array([0, 1, 2, 5, 8, 11])
##FALAO PATH
__C.CHARTLET.FL_DRESS_PATH = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/python/tietu/falao/dress_1.png'
__C.CHARTLET.FL_CAP_PATH = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/python/tietu/falao/toutao_1.png' 
__C.CHARTLET.FL_COLLAR_PATH = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/python/tietu/falao/collar_1.png'
__C.CHARTLET.FL_DRESS_RHIP = [75, 20]
__C.CHARTLET.FL_DRESS_LHIP = [170, 20]
__C.CHARTLET.FL_DRESS_RKNE = [66, 226]
__C.CHARTLET.FL_CAP_HEAD = [110, 50]
__C.CHARTLET.FL_CAP_RSHO = [8, 140]
__C.CHARTLET.FL_CAP_LSHO = [200, 140]
__C.CHARTLET.FL_JOINTSINDEX = np.array([0, 1, 2, 5, 8, 9, 11])
##CAOQUN PATH
__C.CHARTLET.CQ_DRESS_PATH = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/python/tietu/caoqun/capqun_1.png'
__C.CHARTLET.CQ_BRA_PATH = '/home/test/tangyudi/liuhuawei/caffe-cpm/testing/python/tietu/caoqun/bra_1.png'
__C.CHARTLET.CQ_DRESS_RHIP = [70, 12]
__C.CHARTLET.CQ_DRESS_LHIP = [130, 12]
__C.CHARTLET.CQ_DRESS_RKNE = [64, 100]
__C.CHARTLET.CQ_JOINTSINDEX = np.array([2, 5, 8, 9, 11])



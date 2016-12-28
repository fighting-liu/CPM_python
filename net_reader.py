import sys
caffe_root = "/home/test/tangyudi/liuhuawei/SSD/caffe/" 
# caffe_root = "/home/test/tangyudi/liuhuawei/caffe-cpm/caffe/" 
sys.path.insert(0, caffe_root + 'python')
import caffe
from config import cfg

deployFile_person = cfg['deployFile_person']
caffemodel_person = cfg['caffemodel_person']
deployFile_cpm = cfg['deployFile_cpm'] 
caffemodel_cpm = cfg['caffemodel_cpm']   
USE_GPU = cfg['use_gpu']
device = cfg['GPUdeviceNumber']

def load_net():
    if USE_GPU: 
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    caffe.set_device(device) 
        
    ## load the dectection net and preprocess the data
    person_net = caffe.Net(deployFile_person, caffemodel_person, caffe.TEST)
    pose_net = caffe.Net(deployFile_cpm, caffemodel_cpm, caffe.TEST)
    
    return person_net, pose_net



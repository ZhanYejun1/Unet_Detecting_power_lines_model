import colorsys
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Variable

from nets.unet import Unet as unet


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
#--------------------------------------------#
class Unet(object):
    _defaults = {
        # "model_path"        : 'logs/line_Epoch300-Total_Loss0.0943-Val_Loss0.0983.pth',  # 指向训练好的unet权重
        "model_path": 'logs/line_Epoch30-Total_Loss0.1228-Val_Loss0.1353.pth',  # 指向训练好的unet权重
        "model_image_size"  : (512, 512, 3),
        "num_classes"       : 2,
        "cuda"              : True,
        #--------------------------------#
        #   blend参数用于控制是否
        #   让识别结果和原图混合
        #--------------------------------#
        "blend"             : False
    }

    #---------------------------------------------------#
    #   初始化UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.net = unet(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()  # unet(2,3)

        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('{} model loaded.'.format(self.model_path))

        if self.num_classes == 2:
            self.colors = [(255, 255, 255),  (0, 0, 0)]
        elif self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                        for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self ,image, size):  # 对图片进行不失真resize，添加灰条
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image,nw,nh
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        # 进行原始图片的备份
        old_img = copy.deepcopy(image)

        orininal_h = np.array(image).shape[0]  # 计算图片高和宽
        orininal_w = np.array(image).shape[1]
        # 对图片进行不失真resize，添加灰条
        image, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))  #"model_image_size"  : (512, 512, 3),

        images = [np.array(image)/255]  # 对图片进行归一化并加上batchsize维度
        images = np.transpose(images,(0,3,1,2))  # 将通道转到第一维度

        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images =images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1) # 取出每一个像素点所对应的最大概率的种类
            # 对预测结果进行截取，截取掉灰条部分
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        # 判断每一个像素点种类，给每一个像素点赋以特定颜色
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')

        # 将获取到的分割图像转换为image，在进行resize
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        if self.blend:
            image = Image.blend(old_img,image,0.7)  # 将分割图像和原始图像进行混合
        
        return image


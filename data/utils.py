from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

import cv2
from torchvision import transforms
tensor_to_image = transforms.ToPILImage()
import time






#saliency map计算
def saliency_extraction(self, xf_ori):

	xf = xf_ori.clone()
	
	eps = 1e-8
	b=xf.size(0)
	c=xf.size(1)
	h=xf.size(2)
	w=xf.size(3)

	coord = torch.zeros(b, 4)
	coord = coord.cuda()

	saliency = torch.sum(xf, dim=1)*(1.0/(c+eps))

	saliency = saliency.contiguous()
	saliency = saliency.view(b, -1)

	sa_min = torch.min(saliency, dim=1)[0]
	sa_max = torch.max(saliency, dim=1)[0]
	interval = sa_max - sa_min

	sa_min = sa_min.contiguous()
	sa_min = sa_min.view(b, 1)
	sa_min = sa_min.expand(h, w, b, 1)
	sa_min = sa_min.contiguous()
	sa_min = rearrange(sa_min, 'h w b 1 -> b 1 h w')

	interval = interval.contiguous()
	interval = interval.view(b, 1)
	interval = interval.expand(h, w, b, 1)
	interval = interval.contiguous()
	interval = rearrange(interval, 'h w b 1 -> b 1 h w')

	saliency = saliency.contiguous()
	saliency = saliency.view(b, 1, h, w)

	saliency = saliency - sa_min
	saliency = saliency/(interval+eps)

	saliency = torch.clamp(saliency, eps, 1)

	for i in range(b):
		img1 = saliency[i,:,:,:]
		img2 = img1.view(1, h, w)
		img2 = img2*255
		img2 = img2.detach().cpu()
		img2 = img2.numpy()
		mat1 = np.uint8(img2)
		mat1 = mat1.transpose(1,2,0)
		thres, mat2 = cv2.threshold(mat1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		contours, hierarchy = cv2.findContours(mat2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		area = []
		# 找到最大的轮廓
		if len(contours)==0:
			coord[i, 0]=0
			coord[i, 1]=0
			coord[i, 2]=w
			coord[i, 3]=h
		else:

			for k in range(len(contours)):
				area.append(cv2.contourArea(contours[k]))
			max_idx = np.argmax(np.array(area))

			p, q, r, s = cv2.boundingRect(contours[max_idx]) 
			coord[i, 0]=p
			coord[i, 1]=q
			coord[i, 2]=r
			coord[i, 3]=s

		#thres = thres/255.0
		#saliency[i,:,:,:] = torch.where(saliency[i,:,:,:] > thres, saliency[i,:,:,:]/saliency[i,:,:,:], saliency[i,:,:,:]-saliency[i,:,:,:])
	
	coord = coord.detach()

	return coord, coord




#以下实验已做，表明得到的注意力权重是对的，找到的视觉对象大致是准的
if image_ori is not None:
    _, part_inx, attention_map = self.part_select(all_weights)

    coord_ori, _ = self.saliency_extraction(attention_map)

    coord = coord_ori.detach().cpu()
    coord = coord.numpy()
    coord = np.uint8(coord)

    inputs_batch_size = image_ori.size(0)

    for i in range(inputs_batch_size):
        a,b,c,d = coord[i]

        p = int(a)
        q = int(b)
        r = int(c)
        s = int(d)

        saliency_figure = image_ori[i,:,:,:].clone()

        #show = saliency_figure[:, 16*int(b):16*(int(b)+int(d)), 16*int(a):16*(int(a)+int(c))]
        show = saliency_figure[:, 16*q:16*(q + s), 16*p:16*(p + r)]

        show = show.unsqueeze(0)
        show = F.interpolate(show, size=[224, 224], mode='bilinear')
        show=show.squeeze(0)

        #展示剪切原图片得到的对象区域
        display = show.clone()
        display = display.detach().cpu()
        image = tensor_to_image(display)
        img_name = './output_images/' + str(time.time())+'.jpg'
        image.save(img_name)
        
        image_ori[i,:,:,:] = show

	
	
	
 
def part_select(self, x):
    length = len(x)
    last_map = x[0]
    for i in range(1, length):
        last_map = torch.matmul(x[i], last_map)

    last_map = last_map[:,:,0,1:-1]

    #print(last_map.shape)

    _, max_inx = last_map.max(2)

    B,C = last_map.size(0),last_map.size(1)
    patch_num = last_map.size(-1)

    H = patch_num ** 0.5
    H = int(H)
    attention_map = last_map.view(B,C,H,H)

    return _, max_inx, attention_map

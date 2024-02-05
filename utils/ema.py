# import torch
import math


class Text_EMA():
    def __init__(self, name, param, args, decay=0.99):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # the flag for whether register ema
        self.flag = True
        self.args = args
        self.register(name, param)

    def register(self, name, param):
        self.shadow[name] = param.data.clone()

    def update(self, name, param):
        assert name in self.shadow
        new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data 
        self.shadow[name] = new_average.clone()
    
    # 带权重更新文本prompt
    def update_weight(self, name, param, weight, h=37):
        assert name in self.shadow
        new_decay = 1 - math.exp(weight/5 - h)
        new_decay = new_decay.reshape(-1, 1, 1).repeat(1, param.data.shape[1], param.data.shape[2])
        # print(new_decay)
        # print(new_decay.shape)
        # print(self.shadow[name].shape)
        new_average = new_decay * self.shadow[name] + (1.0 - new_decay) * param.data 
        # print(self.shadow[name].shape)
        # print(new_average.shape)
        # print(self.shadow[name])
        # print(new_average)
        # input()
        self.shadow[name] = new_average.clone()
    
    # 只更新预测类别的prompt
    def update_one(self, name, param, pred_class):
        assert name in self.shadow
        # print(self.shadow[name].shape)
        # input()
        new_average = self.decay * self.shadow[name][pred_class,:,:] + (1.0 - self.decay) * param.data[pred_class,:,:]
        # print(new_average.shape) 
        # print(self.shadow[name][pred_class,:,:])
        self.shadow[name][pred_class,:,:] = new_average.clone()
        # print(self.shadow[name][pred_class,:,:])
        # input()    
    
    # 带自适应权重更新一个类别的文本prompt    
    def update_one_weight(self, name, param, pred_class, weight, h=37):
        assert name in self.shadow
        # print("weight:")
        # print(weight)
        # input()
        new_decay = math.exp(-weight/h)
        # new_decay = 1 - torch.exp(weight - h)
        # print("new_decay:")
        # print(new_decay)
        new_average = new_decay * self.shadow[name][pred_class,:,:] + (1.0 - new_decay) * param.data[pred_class,:,:]
        self.shadow[name][pred_class,:,:] = new_average.clone()
        # input()

    def apply_shadow(self, name, param, w=0.5):
        rnt = w * self.shadow[name] + (1.0 - w) * param.data 
        return rnt.clone()


class Image_EMA():
    def __init__(self, name, param, args, decay=0.99):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # the flag for whether register ema
        self.flag = True
        self.args = args
        self.register(name, param)

    def register(self, name, param):
        self.shadow[name] = param.data.clone()

    def update(self, name, param):
        assert name in self.shadow
        new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data 
        self.shadow[name] = new_average.clone()
    
    # 带权重更新图片prompt
    def update_weight(self, name, param, weight, h=5000):
        assert name in self.shadow
        new_decay = math.exp(-weight/h)
        # new_decay = new_decay.reshape(-1, 1, 1).repeat(1, param.data.shape[1], param.data.shape[2])
        # print(new_decay)
        # print(new_decay.shape)
        # print(self.shadow[name].shape)
        new_average = new_decay * self.shadow[name] + (1.0 - new_decay) * param.data 
        self.shadow[name] = new_average.clone()
    
    # 只更新预测类别的图片prompt
    def update_one(self, name, param, pred_class):
        assert name in self.shadow
        # print(self.shadow[name].shape)
        # input()
        new_average = self.decay * self.shadow[name][pred_class] + (1.0 - self.decay) * param.data
        # print(new_average.shape) 
        # print(self.shadow[name][pred_class,:,:])
        self.shadow[name][pred_class] = new_average.clone()
        # print(self.shadow[name][pred_class,:,:])
        # input()    
    
    # 带自适应权重更新一个类别的文本prompt    
    def update_one_weight(self, name, param, pred_class, weight, h=37):
        assert name in self.shadow
        # print("weight:")
        # print(weight)
        # print(-weight/h)
        # input()
        new_decay = math.exp(-weight/h)
        # new_decay = 1 - torch.exp(weight - h)
        # print("new_decay:")
        # print(new_decay)
        new_average = new_decay * self.shadow[name][pred_class] + (1.0 - new_decay) * param.data
        self.shadow[name][pred_class,:,:] = new_average.clone()
        # input()

    def apply_shadow(self, name, param, w=0.5):
        rnt = w * self.shadow[name] + (1.0 - w) * param.data 
        return rnt.clone()
    
    def apply_shadow_one(self, name, param, pred_class, w=0.5):
        rnt = w * self.shadow[name][pred_class] + (1.0 - w) * param.data 
        return rnt.clone()


import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import os


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.my_clip_CSTP import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask, tiny_imagenet_mask

from utils.aug_tools import AugTools
from utils.ema import Text_EMA, Image_EMA

from torchvision.datasets import CIFAR100

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs) 
            if j == 0:
                global aug_tools
                aug_tools.cal_clip(output[0])
            selected_idx = None
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)
            aug_tools.cal_aug(output, j)
            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs) 
            output, selected_idx = select_confident_samples(output, args.selection_p)
            aug_tools.cal_trained_aug(output, j)
    output_1 = torch.mean(output, dim=0)
    if args.cocoop:
        return output, pgen_ctx
    return output_1, output

aug_tools = None

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    global aug_tools
    aug_tools = AugTools(args)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)

    


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if args.myclip:
        model = get_coop(args, args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
    model_state = None

    for name, param in model.named_parameters():
        # print(name)
        if args.myclip:
            if name == "image_prompts" and args.image_prompts:
                param.requires_grad_(True)
            elif "prompt_transformer" not in name:
                param.requires_grad_(False)
        elif not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    elif args.myclip:
        trainable_param = []
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        if args.tpt:
            base_transform = transforms.Compose([
                    transforms.Resize(args.resolution, interpolation=BICUBIC),
                    transforms.CenterCrop(args.resolution)])
            # if args.resize_flag is True:
            #     base_transform = transforms.Compose([
            #         transforms.Resize(args.resize, interpolation=BICUBIC),
            #         transforms.CenterCrop(args.resolution)])
            # else:
            #     base_transform = transforms.Compose([
            #         transforms.Resize(args.resolution, interpolation=BICUBIC),
            #         transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>1, args=args)
            batchsize = 1

        print("evaluating: {}".format(set_id))
        if set_id in ['A', 'R', 'K', 'V', 'I', 'C', 'T']:
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        elif args.myclip:
            model.init_text_prompts(classnames)
            print("learnable text: ", args.learnable_text)
            if args.CSTP == 1:
                if args.learnable_text == "a":
                    trainable_param += [model.text_prompts_a]
                    # print("learnable text: ", args.learnable_text)
                    # input()
                elif args.learnable_text == "a+cls":
                    trainable_param += [model.text_prompts_a]
                    trainable_param += [model.text_prompts_class]
                elif args.learnable_text == "S+a+cls+E":
                    trainable_param += [model.text_prompts_S]
                    trainable_param += [model.text_prompts_a]
                    trainable_param += [model.text_prompts_class]
                    trainable_param += [model.text_prompts_E]
                elif args.learnable_text == "all":
                    trainable_param += [model.text_prompts]
            elif args.CSTP == 2:
                if args.learnable_text == "a":
                    trainable_param += [model.CSTP_bvector]
                    trainable_param += [model.text_prompts_a]
            if args.image_prompts:
                trainable_param += [model.image_prompts]
            elif args.prompt_pool:
                trainable_param += [model.prompt_pool.keys]
                trainable_param += [model.prompt_pool.img_prompts]
            optimizer = torch.optim.AdamW(trainable_param, args.lr)
            optim_state = deepcopy(optimizer.state_dict())
        else:
            model.reset_classnames(classnames, args.arch)
        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode, domain_id=args.domain)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    # batch_size=batchsize, shuffle=False,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            
        # 
        if args.text_prompt_ema:
            if args.myclip:
                text_ema = Text_EMA("text_prompts_a", model.text_prompts_a, args, args.text_prompt_ema_decay)
            else:
                text_ema = Text_EMA("prompt_learner.ctx", model.prompt_learner.ctx, args)
        else:
            text_ema = None
        if args.image_prompt_ema == 1 or args.image_prompt_ema == 2:
            image_ema = Image_EMA("image_prompts", model.image_prompts, args, args.image_prompt_ema_decay)
        elif args.image_prompt_ema == 3 or args.image_prompt_ema == 4:
            class_image_prompts = model.image_prompts.unsqueeze(0).repeat(len(classnames), 1, 1, 1, 1)
            image_ema = Image_EMA("image_prompts", class_image_prompts, args, args.image_prompt_ema_decay)
        else:
            image_ema = None

        test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, text_ema, image_ema)
        del val_dataset, val_loader



def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, text_ema=None, image_ema=None):

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            ## TODO
            model.reset()
    end = time.time()

    global aug_tools

    for i, (images, target) in enumerate(val_loader):
        # print(target)
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)

        aug_tools.target = target

        if args.tpt:
            images = torch.cat(images, dim=0)
            
        # reset the tunable prompt to its initial state
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.myclip:
                with torch.no_grad():
                    model.reset_Tclass_prompts()
                    if args.reset_image_prompts:
                        model.reset_image_prompts()
                    if args.share_prompts != 0 and args.reset_share_prompts:
                        model.reset_share_prompts()
            elif args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            if args.text_prompt_ema:
                new_param = text_ema.apply_shadow("text_prompts_a", 
                                                            model.text_prompts_a, 
                                                            w=args.text_prompt_ema_w)
                model.state_dict()["text_prompts_a"].copy_(new_param)
            if args.image_prompt_ema == 1 or args.image_prompt_ema == 2:
                new_param = image_ema.apply_shadow("image_prompts", 
                                                    model.image_prompts, 
                                                    w=args.image_prompt_ema_w)
                model.state_dict()["image_prompts"].copy_(new_param)
            optimizer.load_state_dict(optim_state)
            output_1, output_aug = test_time_tuning(model, images, optimizer, scaler, args)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            output_1, pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

        if args.image_prompt_ema == 3 or args.image_prompt_ema == 4:
            output_tmp = output_1.unsqueeze(0)
            pred_class = output_tmp.argmax(dim=1, keepdim=True).squeeze()
            weight = output_tmp.max(dim=1, keepdim=True)[0].squeeze()
            new_param = image_ema.apply_shadow_one("image_prompts", 
                                                    model.image_prompts,
                                                    pred_class, 
                                                    w=args.image_prompt_ema_w)
            model.state_dict()["image_prompts"].copy_(new_param)
        
        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)       
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(image)

        aug_tools.cal_trained(output)
        aug_tools.cal_acc(output)

        pred_class = output.argmax(dim=1, keepdim=True).squeeze().item()
        weight = output.max(dim=1, keepdim=True)[0].squeeze().item()
        if args.text_prompt_ema:
            if args.text_prompt_ema_weight:
                text_ema.update_weight("text_prompts_a", model.text_prompts_a, output.squeeze(),
                                        args.text_prompt_ema_weight_h)
            elif args.text_prompt_ema_one:
                text_ema.update_one("text_prompts_a", model.text_prompts_a, pred_class)
            elif args.text_prompt_ema_one_weight:
                text_ema.update_one_weight("text_prompts_a", model.text_prompts_a, 
                                            pred_class, weight,
                                            args.text_prompt_ema_one_weight_h)
            else:
                text_ema.update("text_prompts_a", model.text_prompts_a)
        if args.image_prompt_ema == 1:
            image_ema.update("image_prompts", model.image_prompts)
        elif args.image_prompt_ema == 2:
            image_ema.update_weight("image_prompts", model.image_prompts, weight, args.image_prompt_ema_h)
        elif args.image_prompt_ema == 3:
            image_ema.update_one("image_prompts", model.image_prompts, pred_class)
        elif args.image_prompt_ema == 4:
            image_ema.update_one_weight("image_prompts", model.image_prompts, pred_class, weight, args.image_prompt_ema_h)

        if (i+1) % args.print_freq == 0:
            aug_tools.logger_nums()
    aug_tools.logger_nums(end=True)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--myclip', action='store_true', default=False, help="")
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resize', default=256, type=int, help='CLIP image resolution')
    parser.add_argument('--resize_flag', default=False, type=bool, help='CLIP image resolution')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default="a_photo_of_a_CLS", type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--break_m', type=int, default=-1)
    parser.add_argument('--info', type=str, default='debugg')
    # the type of augmentation
    parser.add_argument('--aug_type', type=str, default='default')
    parser.add_argument('--CSTP', type=int, default=1)
    parser.add_argument('--CSTP_N', type=int, default=200)
    parser.add_argument('--text_prompt_ema', action='store_true', default=False)
    parser.add_argument('--text_prompt_ema_weight', action='store_true', default=False)
    parser.add_argument('--text_prompt_ema_weight_h', type=float, default=37)
    parser.add_argument('--text_prompt_ema_one', action='store_true', default=False)
    parser.add_argument('--text_prompt_ema_one_weight', action='store_true', default=False)
    parser.add_argument('--text_prompt_ema_one_weight_h', type=float, default=37)
    parser.add_argument('--text_prompt_ema_w', type=float, default=0.5)
    parser.add_argument('--text_prompt_ema_decay', type=float, default=0.99)
    parser.add_argument('--learnable_text', type=str, default="a")
    
    parser.add_argument('--image_prompts', action='store_true', default=False, help="")
    parser.add_argument('--prefix_tuning', action='store_true', default=True, help="Using Prefix Tuning if True, \
                                                                    Prompt Tuning if False")
    parser.add_argument('--reset_image_prompts', action='store_true', default=False, help="")
    parser.add_argument('--image_prompt_layer', nargs="+", type=int, default=[1])
    
    parser.add_argument('--image_prompt_ema', type=float, default=0, help="")
    parser.add_argument('--image_prompt_ema_decay', type=float, default=0.995, help="")
    parser.add_argument('--image_prompt_ema_w', type=float, default=0.5, help="")
    parser.add_argument('--image_prompt_ema_h', type=float, default=5000, help="")
    parser.add_argument('--share_prompts', type=int , default=0)
    parser.add_argument('--prompt_pool', action='store_true', default=False, help="use prompt pool")
    
    parser.add_argument('--domain', type=str, default="brightness")
    
    
    main()
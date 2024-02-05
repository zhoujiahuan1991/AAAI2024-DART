
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
import os

import numpy as np

# from utils.prompt_pool import PromptPool

from torch.utils.checkpoint import checkpoint

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='../~/.cache/clip'

class ClipTestTimeTuning(nn.Module):
    def __init__(
        self, args, device, classnames,
        criterion='cosine', arch="ViT-L/14"
    ):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.clip = clip
        self.image_encoder = clip.visual
        # self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        self.criterion = criterion
        self.args = args
        self.train_mode = True
        if self.args.image_prompts:
            self.image_prompt_layers = self.args.image_prompt_layer
            if self.args.prefix_tuning:
                if self.args.arch == "ViT-L/14":
                    self.image_prompts = torch.empty(
                    (1, len(self.image_prompt_layers), 2, 1024), 
                    dtype=self.clip.dtype,
                    device="cuda").uniform_(-1, 1).requires_grad_(True)
                else:
                    self.image_prompts = torch.empty(
                        # TODO
                        ### per image uses the same prompt
                        (1, len(self.image_prompt_layers), 2, 768), 
                        ### per image uses different prompt
                        # (self.args.batch_size, len(self.image_prompt_layers), 2, 768), 
                        dtype=self.clip.dtype,
                        device="cuda").uniform_(-1, 1).requires_grad_(True)
            else:
                self.image_prompts = torch.empty(
                    # TODO
                    # (1, 1, 768), 
                    (self.args.batch_size, 1, 768), 
                    dtype=self.clip.dtype,
                    device="cuda").uniform_(-1, 1).requires_grad_(True)
            self.image_prompts = nn.Parameter(self.image_prompts)
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    def reset(self):
        self.reset_Tclass_prompts()
        # print(model.image_prompts.requires_grad)
        if self.args.image_prompts and self.args.reset_image_prompts is True:
            # print(model.image_prompts.requires_grad)
            self.reset_image_prompts()
            # print(model.image_prompts.requires_grad)

    def reset_share_prompts(self):
        self.share_prompts.uniform_(-1, 1).requires_grad_(True)
    
    def reset_image_prompts(self):
        # print("reset image prompts")
        # print(self.image_prompts[0:10])
        self.image_prompts.uniform_(-1, 1).requires_grad_(True)
        
    def reset_Tclass_prompts(self):
        # print("reset Tclass prompts")
        # TODO
        if self.args.CSTP == 1:
            if self.args.learnable_text == "a":
                self.text_prompts_a.copy_(self.text_prompts_a_init)
            elif self.args.learnable_text == "a+cls":
                self.text_prompts_a.copy_(self.text_prompts_a_init)
                self.text_prompts_class.copy_(self.text_prompts_class_init)
            elif self.args.learnable_text == "S+a+cls+E":
                self.text_prompts_S.copy_(self.text_prompts_S_init)
                self.text_prompts_a.copy_(self.text_prompts_a_init)
                self.text_prompts_class.copy_(self.text_prompts_class_init)
                self.text_prompts_E.copy_(self.text_prompts_E_init)
            elif self.args.learnable_text == "all":
                self.text_prompts.copy_(self.text_prompts_init)
        elif self.args.CSTP == 2:
            if self.args.learnable_text == "a":
                self.CSTP_bvector.copy_(self.text_prompts_a_init)
                text_prompts_a = torch.matmul(self.CSTP_weight, self.CSTP_bvector.reshape(self.args.CSTP_N, -1))
                text_prompts_a = text_prompts_a.reshape(len(self.classnames), 4, -1)
                self.text_prompts_a.copy_(text_prompts_a)
        return



    def init_text_prompts(self, classnames):        
        # 如果每个类使用一个单独的文本prompt
        if self.args.CSTP == 1:
            self.classnames = classnames
            # print(self.args.ctx_init)
            # print(self.args.n_ctx)
            ctx_init = self.args.ctx_init.replace("_", " ")
            ctx_init = self.args.ctx_init.replace("-", " ")
            # print(ctx_init)
            # input()
            # x = ['This is a photo of a '+classname for classname in classnames] 
            # p_len = 6
            x = [ctx_init.replace("CLS", classname) for classname in classnames] 
            # print(x)
            # input()
            # x = ['a photo of a'+classname for classname in classnames] 
            p_len = self.args.n_ctx
            # x = ["X" * self.args.text_plen] + x
            # x = ["X " + p for p in x]
            # x = ["X " * self.args.text_plen + p for p in x]
            x = tokenize(x) # [class_num, n_ctx]
            self.tokenized_text = x.detach().clone()
            x = x.to("cuda")
            self.text_prompts_token = x.detach().clone()
            x = self.clip.token_embedding(x).type(self.dtype)  # [class_num, n_ctx, d_model]
            # TODO
            if self.args.learnable_text == "a":
                self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:p_len+1,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:p_len+1,:].detach().clone()
                self.text_prompts_class = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,p_len+2:,:].detach().clone()
            elif self.args.learnable_text == "a+cls":
                self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:5,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:5,:].detach().clone()
                self.text_prompts_class = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_class = nn.Parameter(self.text_prompts_class)
                self.text_prompts_class_init = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,6:,:].detach().clone()
            elif self.args.learnable_text == "S+a+cls+E":
                self.text_prompts_S = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_S = nn.Parameter(self.text_prompts_S)
                self.text_prompts_S_init = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:5,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:5,:].detach().clone()
                self.text_prompts_class = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_class = nn.Parameter(self.text_prompts_class)
                self.text_prompts_class_init = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_E = x[:,6,:].detach().clone().unsqueeze(1)
                self.text_prompts_E = nn.Parameter(self.text_prompts_E)
                self.text_prompts_E_init = x[:,6,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,7:,:].detach().clone()
            elif self.args.learnable_text == "all":
                self.text_prompts = nn.Parameter(x)
                self.text_prompts_init = x.detach().clone()
        del x
        return

    def similarity(self, q, k, topN=1):
        q = nn.functional.normalize(q, dim=-1)  # q shape [batch_size, 512]
        k = nn.functional.normalize(k, dim=-1)  # k shape [pool_size, 512]
        sim = torch.matmul(q, k.T)  # (B, T)
        # if self.args.prompt_penalty == 0 :
        if self.args.prompt_penalty == 0 or self.train_mode == False:
            dist = 1 - sim
        # elif self.args.prompt_penalty == 1 :
        elif self.args.prompt_penalty == 1 and self.train_mode == True:
            prompt_selected_sum = torch.Tensor(self.prompt_selected_sum_train)
            prompt_selected_sum = prompt_selected_sum.to('cuda')
            total = torch.sum(prompt_selected_sum)
            if total == 0:
                freq = prompt_selected_sum / 1
            else:
                freq = prompt_selected_sum / total
            dist = 1 - sim
            # dist = dist + freq * self.args.pool_size * 0.1
            dist = dist + freq * self.args.pool_size * 0.05
            # dist = dist + freq * self.args.pool_size * 0.5
            # dist = dist + freq * torch.exp(-total)
        val, idx = torch.topk(dist, topN, dim=1, largest=False)
        dist_pick = []
        for b in range(idx.shape[0]):
            pick = []
            for i in range(idx.shape[1]):
                pick.append(dist[b][idx[b][i]])
            dist_pick.append(torch.stack(pick))
        dist = torch.stack(dist_pick)
        # print("idx:", idx)
        return dist, idx

    

    def get_text_features(self, prompts=None):
        # TODO
        # print("self.image_prompts.shape")
        # print(self.image_prompts.shape)
        # print("self.text_prompts_a.shape")
        # print(self.text_prompts_a.shape)
        # input()
        if self.args.learnable_text == "a" or self.args.learnable_text == "a+cls":
            x = torch.cat(( self.text_prompts_begin, 
                            self.text_prompts_a,
                            self.text_prompts_class,
                            self.text_prompts_end), dim=1)     # [batch_size, n_ctx, d_model]
            # print(x.shape)
            # input()
        elif self.args.learnable_text == "S+a+cls+E":
            x = torch.cat(( self.text_prompts_S, 
                            self.text_prompts_a,
                            self.text_prompts_class,
                            self.text_prompts_E,
                            self.text_prompts_end), dim=1)
        elif self.args.learnable_text == "all":
            x = self.text_prompts     # [class_num, n_ctx, d_model]   

        # x[:,1:self.args.text_plen+1,:] = prompts.repeat(x.shape[0], 1, 1)
        # a = x[:, 0:1,:]
        # b = prompts.repeat(x.shape[0], 1, 1)
        # c = x[:, self.args.text_plen+1:,:]
        # x = torch.cat((a,b,c), dim=1)
        # x[:,1:self.args.text_plen+1,:] = prompts.repeat(x.shape[0], 1, 1)
        # print(x.shape)  # [class_num, n_ctx, d_model]
        # x = x.repeat(self.args.batch_size,1,1,1)
        class_num, _, dim = x.shape
        # batch_size, class_num, _, dim = x.shape
        # text_prompts = text_prompts.repeat(class_num,1,1,1).permute(1,0,2,3)
        # x[:,:,1:self.args.text_plen+1,:] = text_prompts
        # [batch_size, n_ctx, d_model]
        x = x + self.clip.positional_embedding.type(self.clip.dtype) 
        ## TODO
        if self.args.share_prompts == 1:
            # print(self.share_prompts.shape)
            # input()
            x = torch.cat(( x[:,:6,:], 
                            self.share_prompts.repeat(class_num, 1, 1)[:,:,:512],
                            x[:,6,:].unsqueeze(1),
                            x[:,8:,:]), dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print(x.shape)  # [n_ctx, class_num, d_model]
        x = self.clip.transformer(x)
        # print(x.shape)  # [n_ctx, class_num, d_model]
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print(x.shape)  # [class_num, n_ctx, d_model]
        x = self.clip.ln_final(x).type(self.clip.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.text_prompts_token.argmax(dim=-1)] @ self.clip.text_projection
        return x
    
    def get_image_embedding(self, x):
        x = self.image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # shape = [*, grid ** 2 + 1, width]
        # print(x.shape)
        # print(self.image_encoder.class_embedding.shape)
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + \
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        # print(x.shape)
        # input()
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        return x

    def get_image_features(self, x, prompts=None):
        if prompts is not None:
            x = torch.cat((x[:,0,:].unsqueeze(1), prompts, x[:,1:,:]), dim=1)  # [64, 197+topN*plen, 768]
        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_pre(x)            # x shape [64, 196(197), 768]
        x = x.permute(1, 0, 2)  # NLD -> LND        # x shape [196(197), 64, 768]
        x = self.image_encoder.transformer(x)       # x shape [196(197), 64, 768]
        x = x.permute(1, 0, 2)  # LND -> NLD        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_post(x[:, 0, :])  # x shape [64, 768]
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj         # x shape [64, 512]
        return x
        

    ### use for prefix tuning 
    def get_image_features_prefix(self, x, prompts=None):
        # TODO
        x = self.image_encoder.ln_pre(x)            # x shape [64, 196(197), 768]
        x = x.permute(1, 0, 2)  # NLD -> LND        # x shape [196(197), 64, 768]
        x, dist = self.transformer_forward_prefix(x, prompts)       # x shape [196(197), 64, 768]
        # x = self.clip.visual.transformer(x)       # x shape [196(197), 64, 768]
        x = x.permute(1, 0, 2)  # LND -> NLD        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_post(x[:, 0, :])  # x shape [64, 768]
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj         # x shape [64, 512]
        return x, dist
    
    ### use for prefix tuning
    def transformer_forward_prefix(self, x, prompts=None):
        Transformer = self.clip.visual.transformer
        dist = 0
        for layer in range(Transformer.layers):
            if layer in self.image_prompt_layers:
                idx = self.image_prompt_layers.index(layer)
                prompts_one_layer = prompts[:, idx, :, :] #[batch, , plen, dim]
            else:
                prompts_one_layer = None
            x = self.ResidualAttentionBlock_forward_prefix(x, layer, prompts_one_layer)
        return x, dist


    ### use for prefix tuning
    def ResidualAttentionBlock_forward_prefix(self, x, layer, prompts=None):
        ### Block: ResidualAttentionBlock self
        Block = self.clip.visual.transformer.resblocks[layer]
        # print(Block)
        # input()
        q = Block.ln_1(x)
        Block.attn_mask = Block.attn_mask.to(dtype=q.dtype, device=q.device) if Block.attn_mask is not None else None
        if prompts is None:
            x = x + Block.attn(q, q, q, need_weights=False, attn_mask=Block.attn_mask)[0]
            # attention_return = Block.attn(x, x, x, need_weights=False, attn_mask=Block.attn_mask)[0]
        else:
            prompts = prompts.permute(1, 0, 2)
            # print(prompts.shape)
            # input()
            half = int(prompts.shape[0]/2)
            # print(prompts[:half, :, :].unsqueeze(1).shape)
            # print(prompts[:half, :, :].shape)
            # print(prompts[:half, :, :].unsqueeze(1).shape)
            # print(prompts[:half, :, :].shape)
            # print(q[0,:,:].unsqueeze(0).shape)
            # print(q[1:,:,:].shape)
            # input()
            k = torch.cat([q[0,:,:].unsqueeze(0), prompts[:half, :, :] , q[1:,:,:]], 0)
            v = torch.cat([q[0,:,:].unsqueeze(0), prompts[half:, :, :] , q[1:,:,:]], 0)
            # k = torch.cat([x[:,0,:].unsqueeze(1), prompts[:, :half ,:] , x[:,1:,:]], 1)
            # v = torch.cat([x[:,0,:].unsqueeze(1), prompts[:, half: ,:] , x[:,1:,:]], 1)
            x = x + Block.attn(q, k, v, need_weights=False, attn_mask=Block.attn_mask)[0]
        # x = x + attention_return
        x = x + Block.mlp(Block.ln_2(x))
        return x


    def cluster_features(self, x):
        return self.get_image_features(self.get_image_embedding(x))

    def inference(self, x):
        batch = x.shape[0]
        dist = 0
        with torch.no_grad():
            x = self.get_image_embedding(x) # embedding shape: [batch_size, 197, 768]
        # image_prompts, text_prompts, dist = self.get_prompts(embedding[:,1:,:])
        ### if use image prompts
        if self.args.image_prompts:
            # print(self.image_prompts.repeat(24,1,1).shape)
            # input()
            ### if use Prefix Tuning
            if self.args.prefix_tuning:
                # if self.args.test_sets in ['V', 'K', 'C', 'A']:
                if self.args.test_sets in ['V', 'C']:
                    x, dist = checkpoint(self.get_image_features_prefix, x, self.image_prompts.repeat(batch,1,1,1))
                    # x, dist = self.get_image_features_prefix(x, self.image_prompts.repeat(batch,1,1,1))
                else:
                    x, dist = self.get_image_features_prefix(x, self.image_prompts.repeat(batch,1,1,1))
            else:    
                # TODO
                if batch == 1:
                    x = self.get_image_features(x, self.image_prompts[0].unsqueeze(0))
                else:
                    x = self.get_image_features(x, self.image_prompts)
        else:
            # x = checkpoint(self.get_image_features, x)
            # with torch.no_grad():
            #     x = self.get_image_features(x)
            x = self.get_image_features(x)

        x = x / x.norm(dim=-1, keepdim=True)
        # if self.args.test_sets in ['V', 'K', 'C', 'A']:
        if self.args.test_sets in ['V', 'K', 'C']:
            text_features = checkpoint(self.get_text_features)
        else:
            text_features = self.get_text_features()
        # text_features = self.get_text_features()
        
        # print(text_features.shape)
        # input()
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        # logits per image
        logits = logit_scale * x @ text_features.t()
        # return logits, dist
        return logits
    
    def forward_clip(self, image):
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            self.tokenized_text = self.tokenized_text.to('cuda')
            # print(self.tokenized_text.device)
            text_features = self.clip.encode_text(self.tokenized_text)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.clip.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            # logits_per_text = logits_per_image.t()

            # shape = [global_batch_size, global_batch_size]
            return logits_per_image

    def forward(self, input, train_mode=True):
        return self.inference(input)
    


# get_coop(args, args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
def get_coop(args, clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes
# args, device, classnames, batch_size, 
#         criterion='cosine', arch="ViT-L/14"
    model = ClipTestTimeTuning(args, device, classnames, arch=clip_arch)

    return model


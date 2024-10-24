""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# # @torch.jit.script
# def SeLU(x, gates, ts, ranges, act=None):
#     # print("debug", act)
#     for i in range(len(gates)):
#         seg = ranges[i]-x
#         x += torch.sigmoid(gates[i]) * act[2*i](seg) * ts[2*i] + \
#              (1 - torch.sigmoid(gates[i])) * act[2*i+1](-seg) * ts[2*i+1]
#     return x

# spline
# def Spline(x, ts, ranges, act=None):
#     x = ts[0] + ts[1]*x
#     for i in range(len(ranges)):
#         x += act[i](x - ranges[i]) * ts[i+2]
#     return x

# residual learning 会比较容易！！！！！！！

# s-relu/s-gelu
# def SeLU(x, ranges, ts, acts):
#     x = x + acts[0](ranges[0] - x) * ts[0] + acts[1](x - ranges[1]) * ts[1]
#     return x



# def SeLU(x, ranges, ts, acts):
#     x = x + acts[0](ranges[0] - x) * ts[0] + acts[1](x - ranges[1]) * ts[1]
#     return x


# def SeLU(x, ranges, ts, acts):
#     x = x + acts[0](ranges[0] - x) * ts[0] + acts[1](x - ranges[1]) * ts[1]
#     return x

#

@torch.jit.script
def SeLU(x, ranges, ts):
# for chebyshev
# def SeLU(x, ts):
    # r1 = ranges[0]
    # r2 = ranges[1]
    # r3 = ranges[2]
    # r4 = ranges[3]
    #
    # # x = x + max(ranges[0] - x, 0) * ts[0] + max(x - ranges[1], 0) * ts[1]
    # # equivalent by just change ts[0] sign
    # # 80.71%!!! best until now
    # # x = x + torch.minimum(x - r1, torch.zeros(1).cuda())**2 * ts[0] + torch.maximum(x - r2, torch.zeros(1).cuda())**2 * ts[1]
    # # polynomial
    # s1 = torch.minimum(x - r1, torch.zeros(1).cuda())
    # s2 = torch.maximum(x - r2, torch.zeros(1).cuda())
    # s3 = torch.minimum(x - r3, torch.zeros(1).cuda())**2
    # s4 = torch.maximum(x - r4, torch.zeros(1).cuda())**2
    # # x = x + s1**2 * ts[0] + s1 * ts[1] + s2**2 * ts[2] + s2 * ts[3]
    # x = x + s1 * ts[0] + s2 * ts[1] \
    #     + s3 * ts[2] + s4 * ts[3]

    # try restricting ts to -1, 1
    # ts = torch.tanh(ts)
    # put tensor to gpu using to_device() or .cuda() is very slow!! avoid such operations
    # x = x + ts[0]*torch.minimum(x - ranges[0], torch.zeros(1).cuda()) + ts[1]* \
    # torch.maximum(x - ranges[1], torch.zeros(1).cuda()) \
    # + ts[2]*torch.minimum(x - ranges[2], torch.zeros(1).cuda())**2 \
    # + ts[3]*torch.maximum(x - ranges[3], torch.zeros(1).cuda())**2

    # this will raise error, using int and tensor not work for minimum
    # x = x + ts[0]*torch.minimum(x - ranges[0], 0) + ts[1]* \
    # torch.maximum(x - ranges[1], 0) \
    # + ts[2]*torch.minimum(x - ranges[2], 0)**2 \
    # + ts[3]*torch.maximum(x - ranges[3], 0)**2

    # use torch.relu implementation, relu is written in c, which is fast!!!!
    x = x + ts[0]*torch.relu(ranges[0] - x) + ts[1]*torch.relu(x - ranges[1]) \
         + ts[2]*torch.relu(ranges[2] - x) **2 + ts[3]*torch.relu(x - ranges[3])**2


    # if it's sparse in the lefthand side, there will be a problem for cross-entropy loss, because no
    # loglikehood is available
    # x = torch.relu(x + ts[1]*torch.relu(x - ranges[1]) + ts[2]*torch.relu(x - ranges[2])**2 - ranges[0])


    # x = x + ts[0]*torch.nn.functional.softplus(ranges[0]-x) + ts[1]*torch.nn.functional.softplus(ranges[1]-x)





    # not good
    # x = x + torch.minimum(x - r1, torch.zeros(1).cuda()) * ts[0] + torch.maximum(x - r2, torch.zeros(1).cuda()) * ts[1] + torch.maximum(torch.minimum(x - r2, torch.zeros(1).cuda()) + r2 - r1, torch.zeros(1).cuda()) * ts[2]
    # to prevent overflow due to exponential in softmax, substact x with its maximum in the softmax dim, it will not change the value of softmax

    # although deit-b returns no error without amp, but training doesn't improve accuracy, so try this
    # clip grad doesn't help solve this problem, accuracy is below 10% after 100 epochs
    # clamp between -20 and 20 doesn't work either, try -10, 10, doesn't work either
    # return torch.clamp(x, min=-10, max=10)
    return x

#这个可以作为验证我的idea, no symmetric的temperature！！！
# def SeLU(x, ranges, ts, acts):
#     x = ts[0]*min(x,ranges[0]), ts[1]*max(x, ranges[1])
#     return x

class Code(nn.Module):
    def __init__(self, numbers):
        super().__init__()
        self.numbers = numbers
        self.codes = nn.Parameter(torch.arange(0., 1., 1/numbers).unsqueeze(0).unsqueeze(0).unsqueeze(0), requires_grad=True)
    def forward(self, x):
        # apply relu on x directly is much worse
        #x = torch.relu(x)
        x_ = x.detach().clone()
        # training speed super slow not because of this no_grad, due to indexing, because batch sizes, sequence length are all included
        with torch.no_grad():
          x_ = torch.relu(x_)
          ind_l = torch.minimum(x_//1, torch.tensor(self.numbers)-1).long()
          ind_r = torch.minimum(x_//1+1, torch.tensor(self.numbers)-1).long()
        #indexing speed is too slow
        #return self.codes[ind]*(1-x+ind) + self.codes[ind+1]*(x-ind)
        # the framework doesn’t know that writes are non-overlapping and intermediate states are discardable. Then you have a long chain of autograd nodes to be processed during backward().
        # gather not work with source tensor shape smaller than index tensor, can replicate and gather, try
        # torch.take is much faster than indexing!! from over 1day to 35 min per epoch
        #return torch.take(self.codes, ind_l)*(1-x+ind_l) + torch.take(self.codes, ind_r)*(x-ind_l)
        B, H, V, _ = x.shape
        codes = self.codes.repeat(B, H, V, 1)
        # gather is fast, it can be further optimized for speed, e.g., assgin the two successive parameters together first
        # then i just need to gather from a large index tensor once
        return torch.gather(codes, -1, ind_l)*(1-x+ind_l) + torch.gather(codes, -1, ind_r)*(x-ind_l)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # this repo actually use bias=True
        # print("qkv bias", qkv_bias)
        # for the pretrained deit model from timm, no need now since there's pretained model in the github homepage
        # self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.range1 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # self.range2 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # # self.range3 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # self.t1 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # self.t2 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # self.t3 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # self.t4 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # self.t5 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # self.t6 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # trunc_normal_(self.t1, std=.02)
        # trunc_normal_(self.t2, std=.02)
        # trunc_normal_(self.t3, std=.02)
        # trunc_normal_(self.t4, std=.02)
        # trunc_normal_(self.t5, std=.02)
        # trunc_normal_(self.t6, std=.02)
        # self.gate1 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # self.gate2 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # # self.gate3 = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # #
        #self.ranges = nn.Parameter(torch.zeros((4)), requires_grad=True)
        # # # never use biases! not good in any form
        # # # self.bs = nn.Parameter(torch.zeros((2)), requires_grad=True)
        # # # self.acts = nn.ModuleList([nn.ReLU() for _ in range(2)])
        # # # # self.a = nn.Parameter(torch.ones((1)), requires_grad=True)
        # # # # # # self.b = nn.Parameter(torch.zeros((1)), requires_grad=True)
        # # # #
        # # #
        #self.ts = nn.Parameter(torch.zeros((4)), requires_grad=True)
        # self.temp = nn.Parameter(torch.ones((1)), requires_grad=True)
        # self.thresholds = nn.Parameter(torch.ones((2)), requires_grad=True)

        # maybe it's better to use zero init, because then the raw attention score is unchanged in the beginning
        # removed after 21.09.2023, recovered from 02.01.2024
        #trunc_normal_(self.ts, std=.02)
        #trunc_normal_(self.ranges, std=.02)

        # self.gates = nn.Parameter(torch.zeros((4)), requires_grad=True)
        # trunc_normal_(self.gates, std=.02)
        # inplace=true will change seg in SeLU every time, leading to unwanted bug!!!!
        # self.acts = nn.ModuleList([nn.ReLU(inplace=False) for _ in range(8)])
        #self.code = Code(100)



        


        # for output12, headwise, worse than shared between heads


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # cosine attention not good no matter with or without scale
        # attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))*self.scale_learnable

        # normalize to angle becomes worse
        # attn = (q/torch.norm(q, dim=-1, keepdim=True)) @ (k/torch.norm(k, dim=-1, keepdim=True)).transpose(-2, -1)
        # problem, sigmoid maximum 1, so the larger range might have problem
        # we could increase the range by scaling it, but again, the range of negative and positive x becomes imbalanced
        # with torch.cuda.amp.autocast(enabled=False):
            # print("max", attn.max(-1)[0])
            # print("min", attn.min(-1)[0])
            # print("sum", attn.sum())

            # current
            # attn = ((1 - self.ratio)*torch.relu(torch.sign(attn - self.range)) + self.ratio)*attn

            # worse than linear one
            # ratio = torch.relu(torch.sign(attn - self.range))
            # left = -attn**2+(1+2*self.range)*attn - self.range**2
            # attn = left*(1-ratio) + attn*ratio

            # # new worse!
            # ratio = torch.relu(torch.sign(attn - self.range))
            # attn = attn - (1-ratio)*self.margin

            # current
            # attn = attn - torch.relu(self.range - attn) * self.margin

            # new, checked, t2 are mostly minus, so the network would like the right slope to be flat
            # t1 are larger than t2
            # both t1 and t2 decreases as the layer number increases
            # for output10
            # attn = attn - torch.relu(self.range - attn) * self.t1 + torch.relu(attn - self.range) * self.t2

            # change the values of range, t1 and t2


            # for output11 very bad!!!! much worse
            # attn = attn - torch.relu(self.range - attn) - torch.relu(attn - self.range)
            # rerun with + right half
            # also very bad!!!! - similar to +
            # attn = attn - torch.relu(self.range - attn) + torch.relu(attn - self.range)

            # new output11, best until now 89.34%, shared between heads and with t1, t2
            # use gap to replace cls, reaches 80.53%
            # attn = attn + torch.relu(self.range1 - attn) * self.t1 + torch.relu(attn - self.range2) * self.t2
            # attn = attn + torch.relu(self.ranges[0] - attn) * self.ts[0] + torch.relu(attn - self.ranges[1]) * self.ts[1]

        self.score = attn
        #attn = SeLU(attn, self.ranges, self.ts)
        # attn = self.temp * attn
        # for chebyshev
        # attn = SeLU(attn, self.ts)



            # now move self.t1 and self.t2 inside, to enable autoflip
            # but this is not enough, because it can not learn the signs after relu, it's always add, but no chance to substract
            # attn = attn + self.t3*torch.relu((self.range1 - attn) * self.t1) + torch.relu((attn - self.range2) * self.t2)*self.t4

            # autoflip corrected 80.25% comparing 80.34%
            # try with only sign outside and with gap, 80.41%
            # attn = attn + torch.sign(self.t3) * torch.relu((self.range1 - attn) * self.t1) + torch.relu(
            #     (self.range2-attn) * self.t2) * torch.sign(self.t4)

            # pre = attn
            # # # add tanh reached 80.61!!!! best ever
            # but the zero initialization causes the gradient to be 0 forever, due to the multiplication of two zero parameters

            # corrected now, check
            # attn = attn + torch.tanh(self.t3) * torch.relu((self.range1 - attn) * self.t1) + torch.relu(
            #     (self.range2-attn) * self.t2) * torch.tanh(self.t4)
            # print(f"diff: {(attn-pre).sum().cpu().item()}")

            # add three components instead of two
            # attn = attn + torch.tanh(self.t3) * torch.relu((self.range1 - attn) * self.t1) + torch.relu(
            #     (self.range2-attn) * self.t2) * torch.tanh(self.t4) + torch.relu(
            #     (self.range3-attn) * self.t5) * torch.tanh(self.t6)

            # 出去tanh部分，会不会等价于linear layer？？？？


            # self.range: (b, h, n, c) -> (b, h, n, 1)
            # # best conifg, add query and key dependent range!!, 80.56, not improved
            # attn = attn + torch.tanh(self.t3) * torch.relu((self.range1(q) - attn) * self.t1) + torch.relu(
            #     (self.range2(k).transpose(-1, -2) - attn) * self.t2) * torch.tanh(self.t4)

            # qk dependent range linear mapping, 80.56

            # try only q dependent range, non linear mapping (linear mapping no improvment, stopped)
            # non linear still no improvement, stopped
            # r = torch.relu(self.map(q))
            # attn = attn + torch.tanh(self.t3) * torch.relu((self.range1(r) - attn) * self.t1) + torch.relu(
            #     (self.range2(r) - attn) * self.t2) * torch.tanh(self.t4)

            # # add tanh inside
            # attn = attn + self.t3 * torch.relu((self.range1 - attn) * torch.tanh(self.t1)) + torch.relu(
            #     (self.range2-attn) * torch.tanh(self.t2)) * self.t4

            # gate, 80.44%, need to rerun too,

            # random init gate 80.42%
            # attn = Spline(attn, self.ts, self.ranges, self.acts)


            # not good at all
            # # for range1<sign<range2, we can use function composition
            # attn = attn + torch.relu(self.range1 - attn) * self.t1 + torch.relu(attn - self.range1)*self.t2
            #
            # # to guarantee that range2 is larger than range1, simply use torch.abs(self.range2) + range1
            # stopped, not good
            # range2 = self.range1 + torch.abs(self.range2)
            # attn = torch.relu(attn - range2)*self.t3

            # or a more elegant solution, decide to adjust left-hand side or right-hand side based on whether range2 and range1 is larger
            # it seems this lead to much worse results at the same epochs
            # 31% vs. 47% at 19 epochs, stopped
            # attn = torch.relu((self.range2 - attn)*torch.sign(self.range1 - self.range2))*self.t3

            # theoretically, we don't need too many segments, because the other part of the model will learn to exploit the most
            # appropriate interval by mapping raw attn scores there

















            #1) we could try w/o flipping, multiply a learnable scaler inside relu, then not only the slope but also the
            # direction is also under control!!!!!!
            #2) ablation of the number of components


            # now try headwise output12, worse than shared
            # attn = attn + torch.relu(self.range1 - attn)* self.t1 + torch.relu(attn - self.range2) * self.t2



            # next rewrite softmax
            # let e**x directly equal to 0, if x < threshold


            # next, headwise!!!!

            # next, predicted range

            # from each patch, mlp maps to eigen-attention

            # for visualization
            # self.score = attn

            ##

            ## temperature
            # attn = attn
        attn = attn.softmax(-1)
        self.attn = attn

            # not work
            # attn = torch.sigmoid(attn)

            # not work
            # attn = torch.exp(attn)/torch.exp(torch.abs(self.a))

            # attn = torch.exp(attn)
            # attn = attn/attn.max(dim=-1, keepdim=True)[0]


        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self.h = x
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            "use naive patchembe[d"
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        print(f"{num_heads} heads")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


        # # # # segmented relu
        #self.ranges = nn.Parameter(torch.zeros((4)), requires_grad=True)
        # self.acts = nn.ModuleList([nn.ReLU() for _ in range(2)])
        #self.ts = nn.Parameter(torch.zeros((4)), requires_grad=True)
        # self.temp = nn.Parameter(torch.ones((1)), requires_grad=True)
        # self.thresholds = nn.Parameter(torch.ones((2)), requires_grad=True)

        # removed after 21.09.2023, recovered from 02.01.2024
        #trunc_normal_(self.ts, std=.02)
        #trunc_normal_(self.ranges, std=.02)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            # x = blk(x)
            x = checkpoint.checkpoint(blk, x)

        x = self.norm(x)
        return x[:, 0]
        #return x.mean(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        # with torch.cuda.amp.autocast(enabled=False):
        #self.feature = x
        #x_bar = torch.mean(x, dim=1, keepdim=True)
        #sigma_x = torch.sqrt(torch.mean((x-x_bar)**2, dim=1, keepdim=True))
        #epsilon = 1e-8
        #sigma_x = torch.clamp(sigma_x, min=epsilon)
        #x = (x-x_bar)/sigma_x

        #x = SeLU(x, self.ranges, self.ts)
        # x = self.temp*x
        # for chebyshev
        # x = SeLU(x, self.ts)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch16_224']
    return model


@register_model
def vit_huge_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch32_384']
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet26d_224']
    return model


@register_model
def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet50d_s3_224']
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet26d_224']
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet50d_224']
    return model

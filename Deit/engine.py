# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import scipy
import math
import sys
from matplotlib.ticker import FormatStrFormatter
from typing import Iterable, Optional
import matplotlib.cm as cm
import torch
from PIL import Image, ImageDraw
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torchvision
from losses import DistillationLoss
import utils
import numpy as np
import time
import subprocess


def ece_score(py, y_test, n_bins=10):
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)

    # print(py_value)
    # print(y_test)

    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000
    start_time = time.time()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # not useful for ddp
        #end_time = time.time()
        #print(end_time - start_time, "loading")
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        # since TF32 is used by default on A100, which is optimized for tensor cores, so there's not too much improvement using amp on A100            
        

        #top2_targets, top2_inds = targets.topk(2, -1)
        #targets = torch.nn.functional.one_hot(targets, 1000) 
        # comment out for checking data loading speed
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(samples)
            # subprocess.run(["nvidia-smi"])
            # with torch.cuda.amp.autocast(enabled=False):
            
            # modification top 1 loss
            #top1_zn = torch.gather(outputs, -1, targets.topk(1, -1)[0])
            top1_zc = outputs.topk(1, -1)[0]
            reg = top1_zc - outputs.mean(-1, keepdim=True)
            
            lam = 0.1 + epoch*0.1/299
            loss = criterion(samples, outputs, targets) + lam*reg.mean()

        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        
        # commented out to test data loading time
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #start_time = time.time()
        #print(start_time-end_time, "computing")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def debug_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets, paths in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        print(paths)

        pass

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# # spline
# def Spline(x, ts, ranges, act=None):
#     x = ts[0] + ts[1]*x
#     for i in range(len(ranges)):
#         x += act[i](x - ranges[i]) * ts[i+2]
#     return x

# sigmoid gates version
# def SeLU(x, gates, ts, ranges, act=None):
#     # numpy array is mutable object
#     # print("input", x)
#     y = 0
#     for i in range(len(gates)):
#         seg = ranges[i]-x
#         y += torch.sigmoid(gates[i]) * act[2*i](seg) * ts[2*i] + \
#              (1 - torch.sigmoid(gates[i])) * act[2*i+1](-seg) * ts[2*i+1]
#     # print("output", x)
#     return y

# def SeLU(x, ranges, ts, acts):
#     x = x + acts[0](ranges[0] - x) * ts[0] + acts[1](x - ranges[1]) * ts[1]
#     return x


def SeLU(x, ranges, ts):
    r1 = ranges[0]
    r2 = ranges[1]
    r3 = ranges[2]
    r4 = ranges[3]

    # x = x + max(ranges[0] - x, 0) * ts[0] + max(x - ranges[1], 0) * ts[1]
    # equivalent by just change ts[0] sign
    # 80.71%!!! best until now
    # x = x + torch.minimum(x - r1, torch.zeros(1).cuda())**2 * ts[0] + torch.maximum(x - r2, torch.zeros(1).cuda())**2 * ts[1]
    # polynomial
    s1 = torch.minimum(x - r1, torch.zeros(1))
    s2 = torch.maximum(x - r2, torch.zeros(1))
    s3 = torch.minimum(x - r3, torch.zeros(1))**2
    s4 = torch.maximum(x - r4, torch.zeros(1))**2
    # x = x + s1**2 * ts[0] + s1 * ts[1] + s2**2 * ts[2] + s2 * ts[3]
    x = x + s1 * ts[0] + s2 * ts[1] + s3 * ts[2] + s4 * ts[3]


    # not good
    # x = x + torch.minimum(x - r1, torch.zeros(1).cuda()) * ts[0] + torch.maximum(x - r2, torch.zeros(1).cuda()) * ts[1] + torch.maximum(torch.minimum(x - r2, torch.zeros(1).cuda()) + r2 - r1, torch.zeros(1).cuda()) * ts[2]
    return x

@torch.no_grad()
def evaluate(data_loader, model, device, epoch=300):
    show_plot = False
    total_images = 640

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if show_plot:
        import matplotlib.pyplot as plt
        import numpy as np
        #
        n = len(model.module.blocks)
        # n = len(model.blocks)

        colors = plt.cm.cool(np.linspace(0, 1, n))
        # for older implementations before 3 segments
        # range1 = []
        # range2 = []

        ranges = []
        ts = []
        y1 = []
        y2 = []
        y3 = []
        y4 = []

        print(f"model.module.blocks: {model.module.blocks}")
        # print(f"model.module.blocks: {model.blocks}")

        plt.figure(figsize=(15,12))
        for i, b in enumerate(model.module.blocks):
        # for i, b in enumerate(model.blocks):
            # for older implementations before 3 segments
            # m = []
            # t = []
            # g = []
            # m.extend([b.attn.range1.cpu(), b.attn.range2.cpu()])
            # t.extend([b.attn.t1.cpu(), b.attn.t2.cpu(), b.attn.t3.cpu(), b.attn.t4.cpu()])
            # g.extend([b.attn.gate1.cpu(), b.attn.gate2.cpu()])
            # range1.append(b.attn.range1.cpu())
            # range2.append(b.attn.range2.cpu())

            m = b.attn.ranges.cpu()
            t = b.attn.ts.cpu()
            # g = b.attn.gates.cpu()
            # a = b.attn.acts.cpu()
            ranges.append(b.attn.ranges.cpu())
            ts.append(t)

            x = torch.linspace(-7, 7, 400)

            # manual setup
            # y = x + torch.relu(m[0] - x) * t[0] + torch.relu(x - m[1]) * t[1]
            # y1.append(range1[i] + torch.relu(range1[i] - m[1]) * t[1])
            # y2.append(range2[i] + torch.relu(m[0] - range2[i]) * t[0])

            # autoflip
            # t.extend([b.attn.t3.cpu(), b.attn.t4.cpu()])
            # y = x + torch.tanh(t[2]) * torch.relu((m[0] - x) * t[0]) + torch.relu(
            #     (m[1] - x) * t[1]) * torch.tanh(t[3])
            # y1.append(range1[i] + torch.relu((m[1] - range1[i]) * t[1]) * torch.tanh(t[3]))
            # y2.append(range2[i] + torch.tanh(t[2]) * torch.relu((m[0] - range2[i]) * t[0]))

            # gate
            # y = x + \
            #         torch.sigmoid(g[0]) * torch.relu(m[0] - x) * t[0] + (
            #             1 - torch.sigmoid(g[0])) * torch.relu(
            #     x - m[0]) * t[1] + torch.sigmoid(g[1]) * torch.relu(m[1] - x)* t[2] + (
            #         1 - torch.sigmoid(g[1])) * torch.relu(x - m[1]) * t[3]
            #
            # y1.append(m[0] + torch.sigmoid(g[1]) * torch.relu(m[1] - m[0])* t[2] + (
            #         1 - torch.sigmoid(g[1])) * torch.relu(m[0] - m[1])* t[3])
            # y2.append(m[1] + \
            #         torch.sigmoid(g[0]) * torch.relu(m[0] - m[1]) * t[0] + (
            #             1 - torch.sigmoid(g[0])) * torch.relu(
            #     m[1] - m[0]) * t[1])

            # 4 segments
            # print(a)
            # y = SeLU(x, g, t, m, a)
            # y1.append(SeLU(m[0], g, t, m, a))
            # y2.append(SeLU(m[1], g, t, m, a))
            # y3.append(SeLU(m[2], g, t, m, a))
            # y4.append(SeLU(m[3], g, t, m, a))

            y = SeLU(x, m, t)
            y1.append(SeLU(m[0], m, t).cpu().item())
            y2.append(SeLU(m[1], m, t).cpu().item())
            y3.append(SeLU(m[2], m, t).cpu().item())
            y4.append(SeLU(m[3], m, t).cpu().item())


            print(f"m: {m}")
            print(f"t: {t}")
            # print(f"g: {g}")
            ax = plt.plot(x, y, color=colors[i])
            # print("x", x)
            # print("y", y)
            # plt.colorbar(
            # )
        # plt.axis('scaled')

        np.savez(f'stats/ranges_epoch_{epoch}.npz', *ranges)
        np.savez(f'stats/ts_epoch_{epoch}.npz', *ts)

        plt.xlim([-9, 9])
        plt.plot(x, x, "--k")
        print(ranges[0].shape)
        print(y1)
        plt.plot([r[0] for r in ranges], y1, "+r")
        plt.plot([r[1] for r in ranges], y2, "*b")
        plt.plot([r[2] for r in ranges], y3, "^g")
        plt.plot([r[3] for r in ranges], y3, ">y")




        plt.xticks([-7 + i * 0.5 for i in range(0, 28)])
        plt.legend([f"layer_{j}" for j in range(len(model.module.blocks))]+["y=x"]+["x=range1"]+["x=range2"] + ["x=range3"] + ["x=range4"])
        # plt.legend([f"layer_{j}" for j in range(len(model.blocks))] + ["y=x"] + ["x=range1"] + ["x=range2"] + [
        #     "x=range3"] + ["x=range4"])
        plt.savefig(f"show/curve.png")
        # print("range")
        # print(m)
        # print("t")
        # print(t)

        y1 = []
        y2 = []
        y3 = []
        y4 = []
        for i, b in enumerate(model.module.blocks):
        # for i, b in enumerate(model.blocks):
            # for implementation before 3 segments
            # m = []
            # t = []
            # m.extend([b.attn.range1.cpu(), b.attn.range2.cpu()])
            # t.extend([b.attn.t1.cpu(), b.attn.t2.cpu(), b.attn.t3.cpu(), b.attn.t4.cpu()])

            # 3 segments
            # g = b.attn.gates.cpu()
            m = b.attn.ranges.cpu()
            t = b.attn.ts.cpu()
            # a = b.attn.acts

            x = torch.linspace(-7, 7, 400)

            # manual setup
            # y = x + torch.relu(m[0] - x) * t[0] + torch.relu(x - m[1]) * t[1]
            # ax = plt.plot(x, torch.exp(y), color=colors[i])
            #
            # y1.append(torch.exp(range1[i] + torch.relu(range1[i] - m[1]) * t[1]))
            # y2.append(torch.exp(range2[i] + torch.relu(m[0] - range2[i])* t[0]))

            # # autoflip
            # t.extend([b.attn.t3.cpu(), b.attn.t4.cpu()])
            # y = x + torch.tanh(t[2]) * torch.relu((m[0] - x) * t[0]) + torch.relu(
            #     (m[1] - x) * t[1]) * torch.tanh(t[3])
            # ax = plt.plot(x, torch.exp(y), color=colors[i])
            # y1.append(torch.exp((range1[i] + torch.relu((m[1] - range1[i]) * t[1]) * torch.tanh(t[3]))))
            # y2.append(torch.exp((range2[i] + torch.tanh(t[2]) * torch.relu((m[0] - range2[i]) * t[0]))))

            # gate
            # y = x + \
            #         torch.sigmoid(g[0]) * torch.relu(m[0] - x)* t[0] + (
            #         1 - torch.sigmoid(g[0])) * torch.relu(
            #     x - m[0]) * t[1] + torch.sigmoid(g[1]) * torch.relu(m[1] - x) * t[2] + (
            #         1 - torch.sigmoid(g[1])) * torch.relu(x - m[1]) * t[3]
            #
            # ax = plt.plot(x, torch.exp(y), color=colors[i])
            #
            # y1.append(torch.exp((m[0] + torch.sigmoid(g[1]) * torch.relu(m[1] - m[0])* t[2] + (
            #         1 - torch.sigmoid(g[1])) * torch.relu(m[0] - m[1]) * t[3])))
            # y2.append(torch.exp((m[1] +
            #         torch.sigmoid(g[0]) * torch.relu(m[0] - m[1])* t[0] + (
            #         1 - torch.sigmoid(g[0])) * torch.relu(
            #     m[1] - m[0]) * t[1])))

            # 3 segments
            # y = SeLU(x, g, t, m, a)
            # y1.append(torch.exp(SeLU(m[0], g, t, m, a)))
            # y2.append(torch.exp(SeLU(m[1], g, t, m, a)))
            # y3.append(torch.exp(SeLU(m[2], g, t, m, a)))
            # y4.append(torch.exp(SeLU(m[3], g, t, m, a)))

            # for 2-dimensional case, when the other entry is equal to 0, it reduces to sigmoid
            # define the other entry z, for every value of x, check 400 different z values, so there will be in total 400x400 pairs of x and z

            points = torch.zeros((400, 400, 2))
            for a in range(-200, 200, 1):
                for b in range(-200, 200, 1):
                    points[199 - a, b + 200][0] = a/200*5
                    points[199 - a, b + 200][1] = b/200*5
            # imshow draw from top to bottom, left to right, just as the matrix definition
            raw_score1 = np.linspace(-5, 5, 11)
            raw_score2 = np.linspace(5, -5, 11)

            # width height

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))

            fig.suptitle("score1")
            plt.ylabel("raw_score1")
            plt.xlabel("raw_score2")


            y = torch.softmax(points, dim=-1)
            y = scipy.stats.entropy(y, axis=-1)
            # it is showing the x axis
            # x and y are int positions for imshow


            ax1.imshow(y)
            # plt.savefig(f"show/softmax_{i}.png")
            ax1.set_title('softmax')
            ax1.set_xticks(list(range(0, 440, 40)), labels=raw_score1)
            ax1.set_xticklabels(raw_score1)
            ax1.set_yticks(list(range(0, 440, 40)), labels=raw_score2)
            ax1.set_yticklabels(raw_score2)
            # the following will make setticks not work!!!! so remove
            # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            # plt.figure(figsize=(12, 12))
            # ax2.ylabel("raw_score1")
            # ax2.xlabel("raw_score2")
            # ax2.title("score1")

            y = torch.softmax(points/0.5, dim=-1)
            y = scipy.stats.entropy(y, axis=-1)
            # it is showing the x axis
            # x and y are int positions for imshow


            ax2.imshow(y)
            # ax2.colorbar()
            # plt.savefig(f"show/softmax_t_half_{i}.png")
            ax2.set_title("softmax_t_0.5")
            ax2.set_xticks(list(range(0, 440, 40)), labels=raw_score1)
            ax2.set_xticklabels(raw_score1)
            ax2.set_yticks(list(range(0, 440, 40)), labels=raw_score2)
            ax2.set_yticklabels(raw_score2)
            # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            # ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            y = torch.softmax(points/2, dim=-1)
            y = scipy.stats.entropy(y, axis=-1)
            # it is showing the x axis
            # x and y are int positions for imshow
            ax3.set_xticks(list(range(0, 440, 40)), labels=raw_score1)
            ax3.set_xticklabels(raw_score1)
            ax3.set_yticks(list(range(0, 440, 40)), labels=raw_score2)
            ax3.set_yticklabels(raw_score2)

            ax3.imshow(y)
            # ax3.colorbar()
            # plt.savefig(f"show/softmax_t_double_{i}.png")


            y = SeLU(points, m, t)
            # plt.figure(figsize=(12, 12))
            # ax4.ylabel("raw_score1")
            # ax4.xlabel("raw_score2")
            ax4.set_title("score1")
            ax3.set_title("softmax_t_2")
            # ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            # ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))


            y = torch.softmax(y, dim=-1)
            y = scipy.stats.entropy(y, axis=-1)
            # it is showing the x axis
            # x and y are int positions for imshow
            # plt.colorbar(ax=ax4)
            im4 = ax4.imshow(y)
            ax4.set_xticks(list(range(0, 440, 40)), labels=raw_score1)
            ax4.set_xticklabels(raw_score1)
            ax4.set_yticks(list(range(0, 440, 40)), labels=raw_score2)
            ax4.set_yticklabels(raw_score2)
            # ax4.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            # ax4.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            fig.subplots_adjust(right=0.95)
            # (left, bottom, width, height)
            cbar_ax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
            fig.colorbar(im4, cax=cbar_ax)
            # ax4.colorbar()

            ax4.set_title("recmax")
            y1.append(torch.exp(SeLU(m[0], m, t)).cpu().item())
            y2.append(torch.exp(SeLU(m[1], m, t)).cpu().item())
            y3.append(torch.exp(SeLU(m[2], m, t)).cpu().item())
            y4.append(torch.exp(SeLU(m[3], m, t)).cpu().item())
            plt.savefig(f"show/map_{i}.png")







            # plt.colorbar(
            # )
        plt.axis('scaled')
        plt.figure(figsize=(15, 12))
        plt.ylim([0, 10])
        plt.xticks([-7 + i * 0.5 for i in range(0, 28)])
        plt.plot(x, torch.exp(x), "--k")
        plt.plot([r[0] for r in ranges], y1, "+r")
        plt.plot([r[1] for r in ranges], y2, "*b")
        plt.plot([r[2] for r in ranges], y3, "^g")
        plt.plot([r[3] for r in ranges], y4, ">y")
        plt.legend([f"layer_{j}" for j in range(len(model.module.blocks))]+["y=e^{x}"])
        plt.legend([f"layer_{j}" for j in range(len(model.module.blocks))] + ["y=e^{x}"])



    # exit()
    #
    # def highlight_grid(image, grid_indexes, grid_size=14):
    #     print(image.max())
    #     IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    #     IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    #     image = Image.fromarray(np.uint8((image*IMAGENET_DEFAULT_STD+IMAGENET_DEFAULT_MEAN)*255), mode='RGB')
    #     if not isinstance(grid_size, tuple):
    #         grid_size = (grid_size, grid_size)
    #
    #     W, H = image.size
    #     h = H / grid_size[0]
    #     w = W / grid_size[1]
    #     image = image.copy()
    #     for grid_index in grid_indexes:
    #         x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
    #         a = ImageDraw.ImageDraw(image)
    #         a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    #     return np.asarray(image)
    # for simplicity and more clear overview, flatten nxn dimensions and head dimensions as well
    if show_plot:
        features = []
        # store each layer separately
        attns_raw = [[] for l in range(len(model.module.blocks))]
        means = [0 for l in range(len(model.module.blocks))]
        mses = [0 for l in range(len(model.module.blocks))]
        maxs = [0 for l in range(len(model.module.blocks))]
        mins = [0 for l in range(len(model.module.blocks))]
        uppers = [0 for l in range(len(model.module.blocks))]
        lowers = [0 for l in range(len(model.module.blocks))]
    counts = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        counts+=1
    #
    #     # compute output
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)

        # # recording raw attn scores
        # for block in model.module.blocks:
        #     attns.extend(block.attn.attn.flatten().cpu().tolist())


        if show_plot:
            features.append(model.module.feature.flatten().cpu())
        #
        #
            raw_scores = [[] for _ in range(len(model.module.blocks))]
        # # # # # #
        # # # # #     torchvision.utils.save_image(images.cpu(), "show/input.png", normalize=True, nrow=images.shape[0])
            for l in range(len(model.module.blocks)):
                attn = model.module.blocks[l].attn.score
                # ranges = model.module.blocks[l].attn.ranges.cpu()
        #
        #         B, H, N, N = attn.shape
        #         attn = attn.float()
        # #
        # #         # for vit with cls token
        #         N = N -1
        #         cls = attn[:, :, -1:, :-1]
        #         attn = attn[:, :, :-1, :-1]
        # # #         # print(attn.shape)
                for h in range(6):
                    # take each head out and reshape b n n to a vector
                    # using list()  and reshape(-1) seems to be extremetly slow, tolist() with flatten() is much faster
                    # maybe on cpu reshape() is very slow
                    # move the flatten before .cpu() will make it faster
                    # raw_scores[l].extend(attn[:, h].flatten().cpu().tolist())
                    attns_raw[l].extend(attn[:, h].flatten().cpu().tolist())
                means[l] += attn.flatten().mean().item()
                mses[l] += (attn.flatten().std()**2).item()
                maxs[l] = max(maxs[l], attn.flatten().max().item())
                mins[l] = min(mins[l], attn.flatten().min().item())
        if show_plot:
            if counts == total_images // 64:
                break

        #
        # # # #
        # # # #
        # # # #
        # # # #         # let softmax attn> 0.5 be red, attn<0.5 be blue
        # # # #
        # # #         for k in range(0, N, 35):
        # # #         # print(attn.shape)
        # # #         # normalized attn is too sparse
        # # #         #     print(attn.shape)
        # # #             out = attn[:, :, k, :].reshape(-1, int(N**(1/2)), int(N**(1/2))).unsqueeze(1)
        # # #             # print(attn[:, k, :].min(dim=-1, keepdim=True)[0].repeat(1, N).shape)
        # # #             min = [attn[:, :, k, :].min(dim=-1, keepdim=True)[0].repeat(1, 1, 1, N*256).reshape(-1, int(N**(1/2))*16, int(N**(1/2))*16).unsqueeze(1)]*2
        # # #             # interpolate to the original size
        # # #             # interleave is the correct way
        # # #             out = out.repeat_interleave(16, -2).repeat_interleave(16, -1)
        # # #
        # # #             # print(out.shape)
        # # #             # print(min[0].shape)
        # # #             # repeat channel dimension, be careful with normalization
        # # #             # use min for extra normalization
        # # #             # out = torch.concat([out] + min, 1)
        # # #
        # # #             ## for blue and red color, softmax attention, no extra normalizaiton
        # # #             # out = torch.concat([torch.relu(out - 0.5)*2] + [torch.zeros_like(out)] + [torch.relu(0.5 - out)*2], dim=1)
        # # #
        # # #             # scale each image individually is important!!!
        # # #             # torchvision.utils.save_image(out, f"show/attn{l}_query{k}.png", nrow=H, normalize=False, scale_each=True)
        # # #             fig, axes = plt.subplots(2, H, figsize=(16, 8))
        # # #             for b in range(0, 8, 4):
        # # #                 for h in range(H):
        # # #                     plt.subplot(2, H, b//4*H + h +1)
        # # #
        # # #                     grid_image = highlight_grid(images[b+1].cpu().permute(1, 2, 0).numpy(), [k], 14)
        # # #
        # # #                     print("grid_image shape", grid_image.shape)
        # # #                     plt.imshow(grid_image)
        # # #                     # by default normalze with min and max value, better for visualization
        # # #                     im = plt.imshow(out.reshape(B, H, 224, 224)[b+1, h], cmap="rainbow", alpha=0.6)
        # # #                     # im = plt.imshow(out.reshape(B, H, 224, 224)[b, h], cmap="rainbow", alpha=0.6,vmin=0, vmax=1)
        # # #                     plt.axis("off")
        # # #             plt.subplots_adjust(
        # # #                             right=0.8,
        # # #                 wspace=0.05, hspace=0.05
        # # #                          )
        # # #
        # # #             cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # # #             fig.colorbar(im, cax=cbar_ax)
        # # #             # plt.colorbar()
        # # #             plt.savefig(f"deit_s/attn{l}_query{k}.png")
        # # #
        # # #
        # # #     exit()
        # #
        # #
        # #
        # #     # visualize modified cls head softmax
        # #     ts = model.ts.cpu()
        # #     ranges = model.ranges.cpu()
        # #     # acts = model.acts.cpu()
        # #     print(ts)
        # #     print(ranges)
        # #
        # #
        # #     x = torch.linspace(-7, 7, 400)
        # #     y = SeLU(x, ranges, ts)
        # #     plt.figure(figsize=(15,12))
        # #     plt.plot(x, y)
        # #     y1, y2 = SeLU(ranges[0], ranges, ts), SeLU(ranges[1], ranges, ts)
        # #     plt.plot(ranges[0], y1, "+r")
        # #     plt.plot(ranges[1], y2, "*b")
        # #     plt.savefig("show/head_softmax")
        # #
        # #     plt.figure(figsize=(15, 12))
        # #     plt.ylim([0, 1000])
        # #     plt.plot(x, torch.exp(y))
        # #     plt.plot(ranges[0], torch.exp(y1), "+r")
        # #     plt.plot(ranges[1], torch.exp(y2), "*b")
        # #     plt.savefig("show/head_softmax_exp")
        # #
        # #     z = torch.sigmoid(y)
        # #     plt.figure()
        # #     plt.plot(x, z)
        # #     plt.plot(x, torch.sigmoid(x))
        # #     plt.plot(x, torch.sigmoid(2 * x))
        # #     plt.savefig("show/2d.png")
        # #




            # exit()







        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # softmax is not applied on the raw output, so it has to be added
        ece = ece_score(torch.softmax(output, -1).cpu().numpy(), target.cpu().numpy())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['ece'].update(ece, n=batch_size)


        # if counts%5==0 and counts>5:
        #     for l in range(len(model.module.blocks)):
        #         plt.figure()
        #         fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        #         for h in range(6):
        #             axes.flatten()[h].boxplot(raw_scores[l][h])
        #         plt.savefig(f"show/stats_layer_{l}_{counts*2048}.png")

            # feature = torch.concat(features, 0)
            # # # feature histogram
            # plt.figure()
            # plt.hist(feature, bins=10)
            # plt.savefig(f"show/feature_hist_{counts*2048}.png")

        # break

    # plt.figure()
    # plt.hist(attns, bins=20)
    # plt.savefig("show/attn histogram")
    # means = [means[l]/counts for l in range(len(model.module.blocks))]
    # stds = [(mses[l] / counts)**0.5 for l in range(len(model.module.blocks))]
    #
    # uppers = [np.percentile(attns_raw[l], 99.5) for l in range(len(model.module.blocks))]
    # lowers = [np.percentile(attns_raw[l], 0.5) for l in range(len(model.module.blocks))]

    # means = np.array(means)
    # stds = np.array(stds)
    # maxs= np.array(maxs)
    # mins= np.array(mins)
    # uppers = np.array(uppers)
    # lowers = np.array(lowers)
    # print(means)
    # print(stds)
    # print(maxs)
    # print(mins)
    # print(uppers)
    # print(lowers)
    # np.savez(f"stats/means_epoch_{epoch}.npz", means)
    # np.savez(f"stats/stds_epoch_{epoch}.npz", stds)
    # np.savez(f"stats/maxs_epoch_{epoch}.npz", maxs)
    # np.savez(f"stats/mins_epoch_{epoch}.npz", mins)
    # np.savez(f"stats/uppers_epoch_{epoch}.npz", uppers)
    # np.savez(f"stats/lowers_epoch_{epoch}.npz", lowers)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(f'ECE {metric_logger.ece.global_avg}')

    # if show_plot:
        # boxplot, too slow even for only 10 batches, give up, not that necessary
        # histgram speed is acceptable
        # plt.figure()
        # plt.figure()
        # fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        # for l in range(len(model.module.blocks)):
        #     # for h in range(6):
        #     print(f"plotting layer {l}")
        #     axes.flatten()[l].hist(attns_raw[l], bins=10)
        #     plt.savefig(f"show/stats_layer_{l}.png")

        # print("start to plot histogram")
        # # hist is very slow if not flatten
        # feature = torch.concat(features, 0)
        # # # feature histogram
        # plt.figure()
        # plt.hist(feature.flatten().numpy(), bins=10)
        # plt.savefig(f"show/feature_hist.png")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sys import argv

from ADMM import compactness_seg_prob_map


def wrap_load(name, path):
    return sio.loadmat(path)[name]


def load_and_config(verbose):
    if verbose:
        print("RIGHTVENT_MRI")

    img = wrap_load('mri', '../Data/mriRV.mat')
    grd_truth = wrap_load('gt', '../Data/gtRV.mat')
    probMap = wrap_load('probMap', '../Data/probMapRV.mat')

    if verbose:
        print(img.shape, grd_truth.shape, probMap.shape)

    # Problem specific parameters
    ParamsADMM = {}
    ParamsADMM['sigma'] = 100
    ParamsADMM['lambda'] = 20000
    ParamsADMM['lambda0'] = 0.5
    ParamsADMM['mul'] = 5000

    # General parameters
    ParamsADMM['imageScale'] = 1
    ParamsADMM['noise'] = 8

    kernel_size = 3
    ParamsADMM['kernel'] = np.ones((kernel_size, kernel_size), np.uint8)
    ParamsADMM['kernel'][kernel_size//2, kernel_size//2] = 0

    if verbose:
        print(ParamsADMM['kernel'])
    ParamsADMM['eps'] = 1e-10

    # Method parameters (Common to all four applications)
    ParamsADMM['mu2'] = 50
    ParamsADMM['mu1Fact'] = 1.01  # Set between 1 and 1.01
    ParamsADMM['mu2Fact'] = 1.01  # Set between 1 and 1.01

    ParamsADMM['solvePCG'] = True  # Use pre-conditioned CG algorithm
    ParamsADMM['maxLoops'] = 1000  # Number of iterations

    # Display options
    ParamsADMM['dispSeg'] = False
    ParamsADMM['dispCost'] = False

    return img, grd_truth, probMap, ParamsADMM


def eval_results(seg, ground):
    TP = np.sum(seg & ground)  # Sum works because those are booleans
    PS = np.sum(seg)
    PG = np.sum(ground)

    dice_index = (2 * TP) / (PS + PG)
    precision = TP / PS
    recall = TP / PG

    return dice_index, precision, recall


def draw_results(img, grd_truth, segCNN, segGCs, segADMM):
    fig, axes = plt.subplots(nrows=1, ncols=4)

    figs = [(grd_truth, "Ground Truth"),
            (segCNN, "seg (CNN)"),
            (segGCs, "Seg (Gcs)"),
            (segADMM, "Seg (ADMM)")]

    for axe,fig in zip(axes.flat, figs):
        axe.imshow(img, cmap="Greys")
        axe.set_title(fig[1])
        axe.contour(fig[0])

    plt.show()


if __name__ == "__main__":
    if len(argv) > 1 and argv[1] == 'v':
        verbose = True
    else:
        verbose = False

    img, grd_truth, probMap, ParamsADMM = load_and_config(verbose)
    if verbose:
        print("Img type:{}, shape:{}".format(img.dtype, img.shape))
        print("Ground truth type: {}, shape:{}".format(grd_truth.dtype, grd_truth.shape))
        print("Probmap type:{}, shape:{}".format(probMap.dtype, probMap.shape))

    segCNN = probMap >= 0.5
    if verbose:
        print("CNN segmentation type:{}".format(segCNN.dtype))
    ParamsADMM['GroundTruth'] = grd_truth

    print("Starting compactness segmentation...")
    segADMM, segGCs, _ = compactness_seg_prob_map(img, probMap, ParamsADMM)

    diceADMM, precisionADMM, recallADMM = eval_results(segADMM, grd_truth)
    diceGCs, precisionGCs, recallGCs = eval_results(segGCs, grd_truth)
    diceCNN, precisionCNN, recallCNN = eval_results(segCNN, grd_truth)

    print(diceADMM, precisionADMM, recallADMM)
    print(diceGCs, precisionGCs, recallGCs)
    print(diceCNN, precisionCNN, recallCNN)

    draw_results(img, grd_truth, segCNN, segGCs, segADMM)

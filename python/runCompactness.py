#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sys import argv

from ADMM import compactness_seg_prob_map


def wrap_load(name, path):
    return sio.loadmat(path)[name]


def load_and_config(choice):
    ParamsADMM = {'imageScale': 1, "noise": 8, "dispSeg": False, "dispCost": False}

    kernel_size = 3
    ParamsADMM['kernel'] = np.ones((kernel_size, kernel_size), np.uint8)
    ParamsADMM['kernel'][kernel_size//2, kernel_size//2] = 0

    ParamsADMM.update({"eps": 1e-10, "mu2":50, "mu1Fact": 1.01, "mu2Fact": 1.01,
                      "solvePCG": True, "maxLoops": 1000})

    print(choice)
    if choice == "RIGHTVENT_MRI":
        img = wrap_load('mri', '../Data/mriRV.mat')
        grd_truth = wrap_load('gt', '../Data/gtRV.mat')
        probMap = wrap_load('probMap', '../Data/probMapRV.mat')

        ParamsADMM.update({"sigma": 100, "lambda": 20000, "lambda0": .5, "mu1": 5000})
    elif choice == "AORTA_MRI":
        img = wrap_load('mri', '../Data/mriAorta.mat')
        grd_truth = wrap_load('gt', '../Data/gtMRIAorta.mat')
        probMap = wrap_load('probMap', '../Data/probMapMRIAorta.mat')

        ParamsADMM.update({"sigma": 25, "lambda": 5000, "lambda0": .5, "mu1": 2000})
    elif choice == "ESOPHAGUS_CT":
        img = wrap_load('ct', '../Data/ctEsophagus.mat')
        grd_truth = wrap_load('gt', '../Data/gtEsophagus.mat')
        probMap = wrap_load('probMap', '../Data/probMapEsophagus.mat')

        ParamsADMM.update({"sigma": 1000, "lambda": 1000, "lambda0": .5, "mu1": 2000})
    elif choice == "AORTA_CT":
        img = wrap_load('ct', '../Data/ctAorta.mat')
        grd_truth = wrap_load('gt', '../Data/gtCTAorta.mat')
        probMap = wrap_load('probMap', '../Data/probMapCTAorta.mat')

        ParamsADMM.update({"sigma": 1000, "lambda": 3000, "lambda0": .5, "mu1": 2000})
    else:
        raise NameError("{} is not a valid choice".format(choice))

    return img, grd_truth, probMap, ParamsADMM


def eval_results(seg, ground):
    t_p = np.sum(seg & ground)  # Sum works because those are booleans
    p_s = np.sum(seg)
    p_g = np.sum(ground)

    dice_index = (2 * t_p) / (p_s + p_g)
    precision = t_p / p_s
    recall = t_p / p_g

    return dice_index, precision, recall


def draw_results(img, grd_truth, segCNN, segGCs, segADMM):
    fig, axes = plt.subplots(nrows=1, ncols=4)

    figs = [(grd_truth, "Ground Truth"),
            (segCNN, "seg (CNN)"),
            (segGCs, "Seg (Gcs)"),
            (segADMM, "Seg (ADMM)")]

    for axe, fig in zip(axes.flat, figs):
        axe.imshow(img, cmap="Greys")
        axe.set_title(fig[1])
        axe.contour(fig[0])

    plt.show()


if __name__ == "__main__":
    choice = "AORTA_CT"
    if len(argv) > 1:
        choice = argv[1]

    img, grd_truth, probMap, ParamsADMM = load_and_config(choice)

    segCNN = probMap >= 0.5
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

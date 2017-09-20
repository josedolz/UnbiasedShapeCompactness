#!/usr/bin/env python3

import graph_tool as gt
import numpy as np
import scipy.io as sio
from sys import argv


def wrap_load(name, path):
    return sio.loadmat(path)[name]

def load_and_config(verbose):
    if verbose:
        print("RIGHTVENT_MRI")

    img = wrap_load('mri', '../Data/mriRV.mat')    
    gt = wrap_load('gt', '../Data/gtRV.mat')    
    probMap = wrap_load('probMap', '../Data/probMapRV.mat')

    if verbose:
        print(img.shape, gt.shape, probMap.shape)

    # Problem specific parameters
    ParamsADMM = {}
    ParamsADMM['sigma'] = 100
    ParamsADMM['lambda'] = 20000
    ParamsADMM['lambda0'] = 0.5
    ParamsADMM['mul'] = 5000

    # General parameters
    ParamsADMM['imageScale'] = 1
    ParamsADMM['noise'] = 8

    kernelSize = 3;
    ParamsADMM['Kernel'] = np.ones((kernelSize, kernelSize))
    ParamsADMM['Kernel'][kernelSize//2, kernelSize//2] = 0
    
    if verbose:
        print(ParamsADMM['Kernel'])
    ParamsADMM['eps'] = 1e-10

    # Method parameters (Common to all four applications)
    ParamsADMM['mu2'] = 50 
    ParamsADMM['mu1Fact'] = 1.01 # Set between 1 and 1.01 
    ParamsADMM['mu2Fact'] = 1.01 # Set between 1 and 1.01 

    ParamsADMM['solvePCG'] = True # Use pre-conditioned CG algorithm
    ParamsADMM['maxLoops'] = 1000 # Number of iterations

    # Display options
    ParamsADMM['dispSeg'] = False
    ParamsADMM['dispCost'] = False

    return img, gt, probMap, ParamsADMM


def compactnessSegProbMap(img, probMap, ParamsADMM):
    '''
    Dummy function for the segmentation
    '''
    return probMap >= 0.5, probMap >= 0.5, 0

def evalResults(Seg, Ground):
    TP = np.sum(Seg & Ground) # Sum works because those are booleans
    PS = np.sum(Seg)
    PG = np.sum(Ground)

    diceIndex = (2 * TP) / (PS + PG)
    precision = TP / PS
    recall = TP / PG

    return diceIndex, precision, recall


if __name__ == "__main__":
    if len(argv) > 1 and argv[1] == 'v':
        verbose = True
    else:
        verbose = False

    img, gt, probMap, ParamsADMM = load_and_config(verbose)
    # print(img.dtype, img.shape)
    # print(gt.dtype, gt.shape)
    # print(probMap.dtype, probMap.shape)


    segCNN = probMap >= 0.5
    # print(CNNSeg.dtype)
    ParamsADMM['GroundTruth'] = gt

    print("Starting compactness segmentation...")
    segADMM, segGCs, _ = compactnessSegProbMap(img, probMap, ParamsADMM)

    diceADMM, precisionADMM, recallADMM = evalResults(segADMM, gt)
    diceGCs, precisionGCs, recallGCs = evalResults(segGCs, gt)
    diceCNN, precisionCNN, recallCNN = evalResults(segCNN, gt)

    print(diceADMM, precisionADMM, recallADMM)
    print(diceGCs, precisionGCs, recallGCs)
    print(diceCNN, precisionCNN, recallCNN)
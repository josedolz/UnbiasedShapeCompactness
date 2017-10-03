# Unbiased Shape Compactness for segmentation
This repository contains the code employed in our work: "Unbiased Shape Compactness for segmentation", which has been accepted at MICCAI 2017 and awarded with the "Students travel award MICCAI".
<br>
<img src="https://github.com/josedolz/UnbiasedShapeCompactness/blob/master/CompactnessResults.png" />
<br>
A version of the paper has been submitted to ArXiv [paper](https://arxiv.org/pdf/1704.08908.pdf)

## Running the code
The code is available only in matlab and python at the moment.

If you use this code for your research, please consider citing the original paper:

- Dolz J, Ben Ayed I, Desrosiers C. "[Unbiased Shape Compactness for segmentation."](https://arxiv.org/pdf/1704.08908.pdf) arXiv preprint arXiv:1704.08908 (2017)

### Matlab
To run it, in the matlab folder, just execute the following function:
```
runCompactness
```

Inside this function you can select which example from the paper you want to reproduce by assigning to the variable example one of the four options (AORTA_MRI, ESOPHAGUS_CT, RIGHTVENT_MRI, AORTA_CT). Compactness parameters are fixed as used in the paper.

### Python
The python implementation is a translation from the Matlab code, and wasn't used in the original paper. Some minor features are missing, but the results are the same. It requires:
* Python 3
* Numpy
* Scipy
* [PyMaxflow](https://github.com/pmneila/PyMaxflow)

You can test it with `./runCompactness.py`.

#### Todo
* Spit admm main loop in several functions
* Add missing sanity tests after Laplacian update
* Argument to select example from command line
    * Fix the mess used for the parameters
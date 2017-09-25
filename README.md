# Unbiased Shape Compactness for segmentation
This repository contains the code employed in our work: "Unbiased Shape Compactness for segmentation", which has been accepted at MICCAI 2017 and awarded with the "Students travel award MICCAI".
<br>
<img src="https://github.com/josedolz/UnbiasedShapeCompactness/blob/master/CompactnessResults.png" />
<br>
A version of the paper has been submitted to ArXiv [paper](https://arxiv.org/pdf/1704.08908.pdf)

## Running the code
The code is available only in matlab at the moment. 

If you use this code for your research, please consider citing the original paper:

- Dolz J, Ben Ayed I, Desrosiers C. "[Unbiased Shape Compactness for segmentation."](https://arxiv.org/pdf/1704.08908.pdf) arXiv preprint arXiv:1704.08908 (2017)

### Matlab
To run it, in the matlab folder, just execute the following function:

```
runCompactness
```
Or alternatively, run in your terminal:
```
make
```
that will launch a headless matlab instance.

Inside this function you can select which example from the paper you want to reproduce by assigning to the variable example one of the four options (AORTA_MRI, ESOPHAGUS_CT, RIGHTVENT_MRI, AORTA_CT). Compactness parameters are fixed as used in the paper.

### Python
The python implementatio is underway. 
At the moment, only the loading, display and computeWeights functions have been implemented. 
To test the current state, type `./runCompactness.py` in your terminal
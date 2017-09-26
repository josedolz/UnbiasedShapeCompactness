addpath(genpath('../Data'));
addpath(genpath('SegmentationCompactnessPrior'))
run('runCompactness.m');
k = waitforbuttonpress;
exit();
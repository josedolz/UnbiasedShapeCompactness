rng(1,'twister');

clc
clear all
close all

%% Load data
global AORTA_MRI; AORTA_MRI = 1;
global ESOPHAGUS_CT; ESOPHAGUS_CT = 2;
global RIGHTVENT_MRI;RIGHTVENT_MRI = 3;
global AORTA_CT;  AORTA_CT = 4;

example = RIGHTVENT_MRI;

switch example
 case 1
  disp(' Runing Aorta MRI example...');
  % Load data
  load('Data/mriAorta.mat'); % CT image
  load('Data/gtMRIAorta.mat'); % Ground truth
  load('Data/probMapMRIAorta.mat'); % CNN result (prob Map)
  img = mri;
  
  % General parameters
  ParamsADMM.sigma = 25; 
  % Specific ADMM parameters
  ParamsADMM.lambda = 5000; % Compactness prior 
  ParamsADMM.lambda0 = 0.5; % MRF binary potential 
  ParamsADMM.mu1 = 2000;
 case 2
  disp(' Runing Esophagus CT example...');
  % Load data
  load('Data/ctEsophagus.mat'); % CT image
  load('Data/gtEsophagus.mat'); % Ground truth
  load('Data/probMapEsophagus.mat'); % CNN result (prob Map)
  img = ct;

  % General parameters
  ParamsADMM.sigma = 1000; 
  % Specific ADMM parameters
  ParamsADMM.lambda = 1000; % Compactness prior 
  ParamsADMM.lambda0 = 0.05; % MRF binary potential
  ParamsADMM.mu1 = 2000;
 case 3
  disp(' Runing Right Ventricle MRI example...');
  % Load data
  load('Data/mriRV.mat'); % CT image
  load('Data/gtRV.mat'); % Ground truth
  load('Data/probMapRV.mat'); % CNN result (prob Map)
  img = mri;
  
  % General parameters
  ParamsADMM.sigma = 100;  
  % Specific ADMM parameters
  ParamsADMM.lambda = 20000; % Compactness prior 
  ParamsADMM.lambda0 = 0.5; % MRF binary potential 
  ParamsADMM.mu1 = 5000;

 otherwise
  disp(' Runing Aorta CT example...');
  % Load data
  load('Data/ctAorta.mat'); % CT image
  load('Data/gtCTAorta.mat'); % Ground truth
  load('Data/probMapCTAorta.mat'); % CNN result (prob Map)
  img = ct;
  
  % General parameters
  ParamsADMM.sigma = 1000;  
  % Specific ADMM parameters
  ParamsADMM.lambda = 3000; % Compactness prior 
  ParamsADMM.lambda0 = 0.5; % MRF binary potential
  ParamsADMM.mu1 = 2000;
end




%% --------- Common ADMM Compactness PARAMETERS ------------------------
%% Parameters
% General parameters
ParamsADMM.imageScale = 1;
ParamsADMM.noise = 8;

kernelSize = 3;
ParamsADMM.Kernel = ones(kernelSize);
ParamsADMM.Kernel((kernelSize+1)/2,(kernelSize+1)/2) = 0;
ParamsADMM.eps = 1e-10;

% Method parameters (Common to all four applications)
ParamsADMM.mu2 = 50; 
ParamsADMM.mu1Fact = 1.01; % Set between 1 and 1.01 
ParamsADMM.mu2Fact = 1.01; % Set between 1 and 1.01 

ParamsADMM.solvePCG = true; % Use pre-conditioned CG algorithm
ParamsADMM.maxLoops = 1000; % Number of iterations

% Display options
ParamsADMM.dispSeg = false;
ParamsADMM.dispCost = false;

options.display = true;


%% Run compactness
% Load BK library
BK_LoadLib;

% Threshold the probability map from the CNN
% NOTE: This probability map can come from any other model (i.e. loglikelihood, chan-vese,etc...)
CNNSeg = zeros(size(probMap));        
CNNSeg(find(probMap>=0.5)) = 1;

ParamsADMM.GroundTruth = gt;

% Segment image
% NOTE: This function will compute segmentation by employing our
% compactness term (SegADMM) and serial BK algorithm (SegGCs)
disp('Starting compactness segmentation...');
[SegADMM,SegGCs,res] = compactnessSegProbMap(img, probMap, ParamsADMM);

% Evaluate results
[diceADMM,precisionADMM,recallADMM] = evalResults(logical(SegADMM), ParamsADMM.GroundTruth);
[diceGCs,precisionGCs,recallGCs]    = evalResults(logical(SegGCs),  ParamsADMM.GroundTruth);
[diceCNN,precisionCNN,recallCNN]    = evalResults(logical(CNNSeg),  ParamsADMM.GroundTruth);

disp(['CNN: ', num2str(diceCNN), ' vs. (CNN + GCs): ', num2str(diceGCs), ' vs. (CNN+GCs+Comp): ',num2str(diceADMM)]);

if options.display
   drawResults(img, gt, CNNSeg, SegGCs, SegADMM); 
end
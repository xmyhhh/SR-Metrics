%% Parameters
% Directory with your results
%%% Make sure the file names are as exactly %%%
%%% as the original ground truth images %%%
function res = evaluate_results(input_dir, GT_dir,image_name,rgb2ycbcr)
%input_dir = fullfile(pwd,'your_results');

% Directory with ground truth images
%GT_dir = fullfile(pwd,'self_validation_HR');

% Number of pixels to shave off image borders when calcualting scores
shave_width = 4;

%% Calculate scores and save
addpath('utils')
scores = calc_scores(input_dir,GT_dir,image_name,shave_width,rgb2ycbcr);
% Saving
%save(strcat(test_name,'.mat'),'scores');

%% Printing results
perceptual_score = (mean([scores.NIQE]) + (10 - mean([scores.Ma]))) / 2;
res=[perceptual_score,mean([scores.Ma]),mean([scores.NIQE]),mean([scores.MSE]),sqrt(mean([scores.MSE])),mean([scores.PSNR]),mean([scores.SSIM]),mean([scores.BRISQUE])];
end

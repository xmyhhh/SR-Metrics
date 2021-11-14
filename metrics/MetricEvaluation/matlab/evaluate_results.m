%% Parameters
% Directory with your results
%%% Make sure the file names are as exactly %%%
%%% as the original ground truth images %%%
function res = evaluate_results(SR_path, GT_path,image_name,rgb2ycbcr,evaluate_Ma)
%input_dir = fullfile(pwd,'your_results');

% Directory with ground truth images
%GT_dir = fullfile(pwd,'self_validation_HR');

% Number of pixels to shave off image borders when calcualting scores
shave_width = 4;

%% Calculate scores and save
addpath('utils');
scores = calc_scores(SR_path,GT_path,image_name,shave_width,rgb2ycbcr,evaluate_Ma);
% Saving
%save(strcat(test_name,'.mat'),'scores');

%% Printing results
if evaluate_Ma
    perceptual_score = (mean([scores.NIQE]) + (10 - mean([scores.Ma]))) / 2;
    res=[mean([scores.NIQE]),mean([scores.BRISQUE]),mean([scores.PIQE]),perceptual_score,mean([scores.Ma])];
else
     res=[mean([scores.NIQE]),mean([scores.BRISQUE]),mean([scores.PIQE])];
end


end

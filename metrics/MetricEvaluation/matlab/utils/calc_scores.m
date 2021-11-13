function [scores] = calc_scores(lr_path,gt_path,image_name,shave_width,rgb2ycbcr)




    
%% Reading and converting images

lr_image = imread(lr_path);
gt_image = imread(gt_path);

if rgb2ycbcr
    lr_image = tools.convert_shave_image(lr_image,shave_width);
    gt_image = tools.convert_shave_image(gt_image,shave_width);
end

if size(lr_image) ~= size(gt_image)
  display(size(lr_image), lr_path)
  display(size(gt_image), gt_path)
end
%% Calculating scores
% scores(ii).name = file_list(ii).name;
scores.MSE = immse(lr_image,gt_image);  % matlab 自带函数
scores.PSNR = psnr(lr_image, gt_image);  % matlab 自带函数

p=genpath('.\MetricEvaluation\utils\srmetric');
addpath(p);
scores.Ma = quality_predict(lr_image);
if ndims(lr_image) == 2
    [scores.SSIM,scores.SSIM_map] = ssim(lr_image, gt_image);
end
if ndims(lr_image) == 3

    for j=1:3
        [ssim_channel(j).SSIM,ssim_channel(j).SSIM_map]= ssim(lr_image(:,:,j), gt_image(:,:,j));
    end 
     scores.SSIM =  mean([ssim_channel.SSIM ]);
     scores.SSIM_map=mean([ssim_channel.SSIM_map]);
end
rmpath(p);
scores.NIQE = niqe.niqe(lr_image);

scores.BRISQUE =brisque.brisquescore(lr_image,image_name);


end



function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end

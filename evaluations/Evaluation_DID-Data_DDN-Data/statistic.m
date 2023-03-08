clear all;
ts =0;
tp =0;
for i=1:1200                         % the number of testing samples DID-Data
%for i=1:1400                          % the number of testing samples DDN-Data
   x_true=im2double(imread(strcat('.\results\',sprintf('%d_GT.png',i))));  % groundtruth 
   x_true = rgb2ycbcr(x_true);
   x_true = x_true(:,:,1); 
   x = im2double(imread(strcat('.\results\',sprintf('%d_DR.png',i))));     %reconstructed image
   x = rgb2ycbcr(x);
   x = x(:,:,1);
   tp= tp+ psnr(x,x_true);
   ts= ts+ssim(x*255,x_true*255);
end
fprintf('psnr=%6.4f, ssim=%6.4f\n',tp/1200,ts/1200)                          % the number of testing samples DID-Data
%fprintf('psnr=%6.4f, ssim=%6.4f\n',tp/1400,ts/1400)                          % the number of testing samples DDN-Data




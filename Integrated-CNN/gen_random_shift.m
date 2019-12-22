clc;
clear;
% 
% imgPath = 'test1_22atg6_1/1/';        % 图像库路径
% outPath = 'test1_22atg6_1shift/1/';
% imgDir  = dir([imgPath '*.csv']); % 遍历所有jpg格式文件
% n = length(imgDir);
% for i = 1:n        % 遍历结构体就可以一一处理图片了
%     mat = importdata([imgPath imgDir(i).name]); %读取每张图片
%     
%     r = randperm(3)-1;
%     shift = r(1);
%     mat1 = [mat(:,1),rand_shift(mat(:,2),shift),mat(:,3:6)];
%     out = [outPath imgDir(i).name];
%     dlmwrite(out,mat1);
% end

imgPath = 'test1_22atg6/1/';        % 图像库路径
outPath = 'test1_22atg6shift/1/';
imgDir  = dir([imgPath '*.csv']); % 遍历所有jpg格式文件
n = length(imgDir);
for i = 1:n        % 遍历结构体就可以一一处理图片了
    mat = importdata([imgPath imgDir(i).name]); %读取每张图片
    
    r = randperm(3)-1;
    shift = r(1);
    mat1 = [mat(:,1),rand_shift(mat(:,2),shift),mat(:,3:6)];
    out = [outPath imgDir(i).name];
    dlmwrite(out,mat1);
end











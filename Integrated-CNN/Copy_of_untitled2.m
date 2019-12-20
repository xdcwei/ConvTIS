
clc;
clear;

imgPath = 'test1_22atg6/0/';        % 图像库路径
outPath = 'test1_22atg6_1/0/';
imgDir  = dir([imgPath '*.csv']); % 遍历所有jpg格式文件
n = length(imgDir);
r = randperm(n);
inds = r(1:floor(n/5));
for i = 1:length(inds)          % 遍历结构体就可以一一处理图片了
    mat = importdata([imgPath imgDir(inds(i)).name]); %读取每张图片
    out = [outPath imgDir(inds(i)).name];
    dlmwrite(out,mat);
end


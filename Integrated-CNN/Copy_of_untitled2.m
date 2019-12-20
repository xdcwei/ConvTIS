
clc;
clear;

imgPath = 'test1_22atg6/0/';        % ͼ���·��
outPath = 'test1_22atg6_1/0/';
imgDir  = dir([imgPath '*.csv']); % ��������jpg��ʽ�ļ�
n = length(imgDir);
r = randperm(n);
inds = r(1:floor(n/5));
for i = 1:length(inds)          % �����ṹ��Ϳ���һһ����ͼƬ��
    mat = importdata([imgPath imgDir(inds(i)).name]); %��ȡÿ��ͼƬ
    out = [outPath imgDir(inds(i)).name];
    dlmwrite(out,mat);
end


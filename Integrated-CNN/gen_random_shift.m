clc;
clear;
% 
% imgPath = 'test1_22atg6_1/1/';        % ͼ���·��
% outPath = 'test1_22atg6_1shift/1/';
% imgDir  = dir([imgPath '*.csv']); % ��������jpg��ʽ�ļ�
% n = length(imgDir);
% for i = 1:n        % �����ṹ��Ϳ���һһ����ͼƬ��
%     mat = importdata([imgPath imgDir(i).name]); %��ȡÿ��ͼƬ
%     
%     r = randperm(3)-1;
%     shift = r(1);
%     mat1 = [mat(:,1),rand_shift(mat(:,2),shift),mat(:,3:6)];
%     out = [outPath imgDir(i).name];
%     dlmwrite(out,mat1);
% end

imgPath = 'test1_22atg6/1/';        % ͼ���·��
outPath = 'test1_22atg6shift/1/';
imgDir  = dir([imgPath '*.csv']); % ��������jpg��ʽ�ļ�
n = length(imgDir);
for i = 1:n        % �����ṹ��Ϳ���һһ����ͼƬ��
    mat = importdata([imgPath imgDir(i).name]); %��ȡÿ��ͼƬ
    
    r = randperm(3)-1;
    shift = r(1);
    mat1 = [mat(:,1),rand_shift(mat(:,2),shift),mat(:,3:6)];
    out = [outPath imgDir(i).name];
    dlmwrite(out,mat1);
end











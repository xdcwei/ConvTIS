function [posArr,posInd,negArr,negInd] = extractCntArr(DNAseq,cdsInd,lws,rws,tp)
%EXTRACTCNTARR 此处显示有关此函数的摘要
%   此处显示详细说明
% tp == 1   表示返回第一位
% tp = 2 返回第二位
% tp = 3 返回第三位
dnaLen = length(DNAseq);
allInd = 1:dnaLen;
posInd = [];
cds = [];
for i = 1:2:length(cdsInd)
   cds = [cds,cdsInd(i):cdsInd(i+1)];
end
for j = tp:3:length(cds)
   posInd = [posInd, cds(j)];
end
negInd = setdiff(allInd,posInd);
sl = blanks(lws);
sr = blanks(rws);
sl(:) = 'N';
sr(:) = 'N';
posArr = repmat(blanks(lws+rws+1),length(posInd),1);
negArr = repmat(blanks(lws+rws+1),length(negInd),1);
dnaExt = [sl,DNAseq,sr];
for i = 1:length(posInd)
    ind = posInd(i);
    posArr(i,:) = dnaExt(ind:ind+lws+rws);
end

for i = 1:length(negInd)
    ind = negInd(i);
    negArr(i,:) = dnaExt(ind:ind+lws+rws);
end

% for i = lws + 1:lws + dnaLen -1 
%     dnaExt(i-lws:i+rws)
% end
end


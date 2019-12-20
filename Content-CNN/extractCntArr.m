function [posArr,posInd,negArr,negInd] = extractCntArr(DNAseq,cdsInd,lws,rws,tp)
%EXTRACTCNTARR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
% tp == 1   ��ʾ���ص�һλ
% tp = 2 ���صڶ�λ
% tp = 3 ���ص���λ
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


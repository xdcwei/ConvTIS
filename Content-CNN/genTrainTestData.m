clc;
clear;
load chr1_22nonrepeat_can_train

ratio = 0.2;
for i = 1207:8000
    i
    if mod(length(chr1_22nonrepeat_can_train(i).CDS),2)
        continue;
    end
    [pArr,pInd,nArr,nInd] = extractCntArr(upper(chr1_22nonrepeat_can_train(i).Sequence),chr1_22nonrepeat_can_train(i).CDS,45,45,1);
    ml = floor(min([size(pArr,1)*ratio,size(nArr,1)]));
    r = randperm(size(nArr,1));
    r2 = randperm(size(pArr,1));
    orthEncodingImg(i,1,pArr(r2(1:ml),:),pInd(r2(1:ml)),1,'train1_22CC\1\');
    orthEncodingImg(i,1,nArr(r(1:ml),:),nInd(r(1:ml)),0,'train1_22CC\0\');
end

load chr1_22nonrepeat_can_test
for i = 1:2831
    i
    if mod(length(chr1_22nonrepeat_can_test(i).CDS),2)
        continue;
    end
    [pArr,pInd,nArr,nInd] = extractCntArr(upper(chr1_22nonrepeat_can_test(i).Sequence),chr1_22nonrepeat_can_test(i).CDS,45,45,1);
    ml = floor(min([size(pArr,1)*ratio,size(nArr,1)]));
    r = randperm(size(nArr,1));
    r2 = randperm(size(pArr,1));
    orthEncodingImg(i,1,pArr(r2(1:ml),:),pInd(r2(1:ml)),1,'test1_22C\1\');
    orthEncodingImg(i,1,nArr(r(1:ml),:),nInd(r(1:ml)),0,'test1_22C\0\');
end


% 
% system('shutdown -s');






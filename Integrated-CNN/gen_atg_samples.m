clc;
clear;

ratio = 5;
dimer = 'ATG';


load chr1_22nonrepeat_can_train
load chr1_22nonrepeat_can_test

for i = 1:8000
    i
    [pos_arr,pos_inds,neg_arr,neg_inds] = fast_extract_seqs(upper(chr1_22nonrepeat_can_train(i).Sequence),chr1_22nonrepeat_can_train(i).CDS,200,200,dimer,ratio);
    orthEncodingImg5(i,5,pos_arr,pos_inds,1,'train1_22atg\1\');
%     ml = floor(min([size(pos_arr,1)*ratio,size(neg_arr,1)]));
%     r = randperm(size(neg_arr,1));
    orthEncodingImg5(i,5,neg_arr,neg_inds,0,'train1_22atg\0\');
end

for i = 1:2831
    i
    [pos_arr,pos_inds,neg_arr,neg_inds] = fast_extract_seqs(upper(chr1_22nonrepeat_can_test(i).Sequence),chr1_22nonrepeat_can_test(i).CDS,200,200,dimer,ratio);
    orthEncodingImg5(i,5,pos_arr,pos_inds,1,'test1_22atg\1\');
%     ml = floor(min([size(pos_arr,1)*ratio,size(neg_arr,1)]));
%     r = randperm(size(neg_arr,1));
    orthEncodingImg5(i,5,neg_arr,neg_inds,0,'test1_22atg\0\');
end









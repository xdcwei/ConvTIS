function [pos_arr,pos_inds,neg_arr,neg_inds] = fast_extract_seqs(seq,cds,lws,rws,dimer,ratio)
%FAST_EXTRACT_SIGNAL_ARR 此处显示有关此函数的摘要
%   此处显示详细说明
len = length(cds);
if strcmp(dimer,'AG')
    pos_inds = cds(3:2:len)-2;
elseif strcmp(dimer,'ATG')
    pos_inds = cds(1);
else
    pos_inds = cds(2:2:len-2)+1;
end
idxs = strfind(seq,dimer);
pos_inds = intersect(idxs,pos_inds);
neg_inds = setdiff(idxs,pos_inds);

lp = length(pos_inds)*ratio;
ln = length(neg_inds);
mnum = min([lp,ln]);
r = randperm(ln);
neg_inds = neg_inds(r(1:mnum));
pos_arr = get_arr_seqs(seq,pos_inds,lws,rws);
neg_arr = get_arr_seqs(seq,neg_inds,lws,rws);
end

function seq_arr = get_arr_seqs(seq,inds,lws,rws)
l = length(inds);
s = blanks(lws+rws+1);
seq_arr = repmat(s,l,1);
for i = 1:l
    seq_arr(i,:) = arr_seq(seq,inds(i),lws,rws);
end
end

function s = arr_seq(seq,ind,lws,rws)
s1 = blanks(lws);
s2 = blanks(rws);
s1(:)= 'N';
s2(:)='N';
seq_ext = [s1,seq,s2];
s = seq_ext(ind:ind+lws+rws);
end

% function score_arr = get_arr_nums(scores,inds,lws,rws)
% l = length(inds);
% score_arr = zeros(l,lws+rws+1);
% for i = 1:l
%     score_arr(i,:) = arr_score(scores,inds(i),lws,rws);
% end
% end
% 
% function score_vec = arr_score(scores,ind,lws,rws)
% scores_ext = [zeros(1,lws),scores,zeros(1,rws)];
% score_vec = scores_ext(ind:ind+lws+rws);
% end

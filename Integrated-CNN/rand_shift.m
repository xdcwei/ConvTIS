function v = rand_shift(vec,s)
%RAND_SHIFT 此处显示有关此函数的摘要
%   此处显示详细说明
l = length(vec);
v = [zeros(s,1);vec(1:l-s)];
end


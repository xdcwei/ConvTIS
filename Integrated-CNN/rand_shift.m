function v = rand_shift(vec,s)
%RAND_SHIFT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
l = length(vec);
v = [zeros(s,1);vec(1:l-s)];
end


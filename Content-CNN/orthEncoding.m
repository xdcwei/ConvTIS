function [mat ] = orthEncoding(seq)
l = length(seq);
mat = zeros(l,4);
for i = 1:l
    s = seq(i);
    if strcmp(s,'A')
        mat(i,:) = [1,0,0,0];
    elseif strcmp(s,'C')
        mat(i,:) = [0,1,0,0];
    elseif strcmp(s,'T')
        mat(i,:) = [0,0,1,0];
    elseif strcmp(s,'G')
        mat(i,:) = [0,0,0,1];
    else
        mat(i,:) = [0,0,0,0];
    end
end
end

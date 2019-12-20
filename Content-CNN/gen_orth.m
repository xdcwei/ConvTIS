clc
clear
load chr1_22nonrepeat_can_test
load chr1_22nonrepeat_can_train

for i = 1:8000
    i
    mat = orthEncoding(upper(chr1_22nonrepeat_can_train(i).Sequence));
    path = strcat('orth_train\',num2str(i),'.csv');
    dlmwrite(path,mat);
end

for i = 1:2831
    i
    mat = orthEncoding(upper(chr1_22nonrepeat_can_test(i).Sequence));
    path = strcat('orth_test\',num2str(i),'.csv');
    dlmwrite(path,mat);
end
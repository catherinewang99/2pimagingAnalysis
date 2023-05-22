crop_num = 414;

L_hit_tmp = L_hit_tmp(1:crop_num);
L_ignore_tmp = L_ignore_tmp(1:crop_num);
L_miss_tmp = L_miss_tmp(1:crop_num);
R_hit_tmp = R_hit_tmp(1:crop_num);
R_ignore_tmp = R_ignore_tmp(1:crop_num);
R_miss_tmp = R_miss_tmp(1:crop_num);

protocol = protocol(1:crop_num);
LickEarly_tmp = LickEarly_tmp(1:crop_num);

delay_duration = delay_duration(1:crop_num);

save 'F:\data\BAYLORCW029\python_behavior\2023_05_21\behavior.mat'
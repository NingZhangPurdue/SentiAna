ratios = [64 32 16 8 4 2]
%diary('log_diff_cate_2')
%diary on
%for i = ratios
%    exp_ETLR(8, 3, i, 10, 5,0);
%end
%
%for i = ratios
%    exp_ETLR(8, 4, i, 10, 5,0);
%end
%diary off

diary('log_diff_cate_8')
diary on
for i = ratios
    exp_ETLR(8, 1, i, 10, 5, 0);
end

for i = ratios
    exp_ETLR(8, 2, i, 10, 5, 0);
end
diary off

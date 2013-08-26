ratio = [64 32 16 8 4 2]

diary('log_diff_cate_sina_2to8')
diary on
for i = ratio
    exp_CDCCET(3,i,5,5,0,0,0,0);
end
diary off

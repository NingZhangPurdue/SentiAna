ratio = [64 32 16 8 4 2]

diary('log_diff_cate_sina_8to2')
diary on
for i = ratio
    exp_CDCCET(1,i,10,5,0,0,0,0);
end
diary off

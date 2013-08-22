diary('log_same_cate')
diary on

ratios = [64, 32, 16, 8, 4, 2]
for i = ratios
    exp_CDET_J(1, i, 10, 5);
end

for i = ratios
    exp_CDET_J(2, i, 10, 5);
end

diary off

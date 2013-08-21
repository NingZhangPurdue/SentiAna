function [bw] = bandwidth_KDE()

data_QQ = load('feaMat_comments_Preprocess_QQEntertainment.txt');
data_Sina = load('feaMat_comments_Preprocess_SinaSociety.txt');

X_qq = sparse(data_QQ(: , 3:end));
X_sina = sparse(data_Sina(: , 3:end));

%X_qq = data_QQ(: , 3:end);
%X_sina = data_Sina(: , 3:end);

X = [X_qq; X_sina];

[N, ~] = size(X)
s = 0
for i = 1:N
    if mod(i,100) == 0
        i
    end
    dist = sqrt(sum((repmat(X(i, : ), N, 1) - X).^ 2, 2));
    dist
    s = s + sum(dist);
end

s/(N^2)

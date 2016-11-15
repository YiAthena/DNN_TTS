function Y = addDelta(X,W)
% Delta feature is added to X based on window matrix W

[D,N] = size(X);
[nwindows,wsize] = size(W);

c = wsize - 1; 

Y = zeros((nwindows+1)*D,N);
Y(1:D,:) = X;
eX = [repmat(X(:,1),[1,c/2]),X,repmat(X(:,N),[1,c/2])]; % [D, N+c]

for n = 1:nwindows
    w = makeDeltawindow(N,W(n,:));
    Y(D*n + (1:D), :) = eX * w;
end

function W = makeDeltawindow(T,wcoef)

wsize = size(wcoef,2);
c = wsize - 1;
nrange = 1:T;

tmp = bsxfun(@plus,(0:c)',nrange);
             row_idx = tmp(:);
             tmp = repmat(nrange,[wsize,1]);
             column_idx = tmp(:);
             data = repmat(wcoef,[1,T]);
             W = sparse(row_idx,column_idx,data,T+c,T);

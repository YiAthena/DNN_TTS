function y = mlpg(X,v,delta)
%
% size(delta) = [n,t]; (e.g) [-0.5 0 0.5; 1 -2 1]
% size(X) : [(n+1)*d , T ]
% v = std .^ 2; ((n+1)*d,1)

[wd,wsize] = size(delta);
wdepth = wd + 1;
c = wsize -1;
[jdim,T] = size(X);

D = repmat(1./v,[1,T]); % [ (n+1)*d , T ]
d = jdim / wdepth;

X = D .* X; 

% make delta window
W = windowmatrix(delta,T);

% boudary condtions
X(d+1:jdim,1:c/2) = 0; X(d+1:jdim,T:-1:T-c/2+1) = 0;
D(d+1:jdim,1:c/2) = 0; D(d+1:jdim,T:-1:T-c/2+1) = 0;

y = zeros(d,T);
for k = 1:d
	tdim = k:d:jdim;
	Xmat = X(tdim,:); % wdepth * T
	Dmat = D(tdim,:); % wdepth * T
	y(k,:) = Xmat(:)' * W / (W'*sparse(1:(wdepth*T),1:(wdepth*T),Dmat(:))*W);
end


function W = windowmatrix(delta,T)

[wd,wsize] = size(delta);
c = wsize - 1;
wdepth = wd + 1;

win =zeros(wdepth,wsize);
win(2:wdepth,:) = delta; win(1,c/2+1) = 1;

% row index
tmp = repmat((1:wdepth)'-1,[1,wsize]);
tmp = bsxfun(@plus,tmp(:),1:wdepth:wdepth*T-1);
row_idx = tmp(:);

% column index
tmp = repmat((1:wsize)-1,[wdepth,1]);
tmp = bsxfun(@plus,tmp(:),1:T);
column_idx = tmp(:);

data = repmat(win(:),[T,1]);
wtmp = sparse(row_idx,column_idx,data,wdepth*T,T+c);
W = sparse(wtmp(:,c/2 + (1:T)));
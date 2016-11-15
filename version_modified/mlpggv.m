function y = mlpggv(X,v,delta,varargin)
%
% size(delta) = [n,t]; (e.g) [-0.5 0 0.5; 1 -2 1]
% size(X) : [(n+1)*d , T ]
% v = std .^ 2; ((n+1)*d,1)

if nargin < 3 || nargin > 5
	error('usage: y = mlpg(X,v,delta) or y = mlpg(X,v,delta,gv)');
end

usegv = false;
if nargin == 4
	gv = varargin{1}; % gv should have mu & Sigma
    usegv =true;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cal mlpg
%m = gmm.mu;
%v = gmm.Sigma;
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if usegv
	y0 = y;
	y0_m = mean(y0,2); % d x 1
    y0_v = mean(y0.^2,2) - y0_m.^2; % d x 1
    mugv = gv.mu;
    vgv = gv.Sigma;
    ratio = sqrt( mugv ./ y0_v);
    y = bsxfun(@plus, bsxfun(@times,ratio,y0), (1 - ratio) .* y0_m);
    gvweight = 1 / T / wdepth;
	alpha = 0.1;
% Configuration (Max num of iteration)
	MIter = 5;
	prevobj = 0;

    for it = 1:MIter

    	dv = zeros(size(y));
    	y_m = mean(y,2);
    	y_v = mean(y.^2,2)-y_m.^2;
    	scale = (y_v - mugv) ./ vgv;
        gvbias = scale .* y_m;
        vdiff = bsxfun(@minus,bsxfun(@times,scale,y),gvbias) * 2 / T;

    	Y = addDelta(y,delta);

    	NlogL=calcPost3(X,Y,v);
    	VlogL=calcPost2(gv,y_v);

		% boundary conditions
        %N(d+1:wdepth*d,1:c/2) = 0; N(d+1:wdepth*d,T:-1:T-c/2+1) = 0;
        %D(d+1:wdepth*d,1:c/2) = 0; D(d+1:wdepth*d,T:-1:T-c/2+1) = 0;

    	obj = gvweight*(-NlogL)-VlogL;
    	for k = 1:d
    		tdim = k:d:wdepth*d;
            Xmat = X(tdim,:); % wdepth * T
            Dmat = D(tdim,:); % wdepth * T
            dv(k,:) = gvweight * (Xmat(:)' * W - y(k,:) * (W'*sparse(1:wdepth*T,1:wdepth*T,Dmat(:))*W));
        end
        if it > 1
        	if obj > prevobj
        		alpha = alpha * 0.5;
        	else
        		alpha = alpha * 1.2;
            end
        end
        dv = dv - vdiff;
        y = y + alpha * dv;
        prevobj = obj;
	end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=calcPost2(gv,y_v)

% Y : d x T
m = gv.mu; % d x 1
v = gv.Sigma; % d x 1
dim = size(m,1);

%v_matrix = diag(v);
%[v_row, v_col]=size(v_matrix);
d1 = bsxfun(@rdivide, y_v.^2, v); % d x T
d2 = bsxfun(@times, y_v,  m ./ v); % d x T
c = sum(m .* m ./ v); % 1 x 1

p = -0.5 * ( sum(sum(d1 - 2*d2)) + c + dim * log(2*pi) + sum(v) );
%p=-1/2*((Y-m)'/v_matrix*(Y-m) +v_matrix*log(2*pi)+log(det(v_matrix)）)；

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=calcPost3(X,Y,v)
% X: d x T
% Y : d x T

normalized_diff = bsxfun(@rdivide, X - Y, sqrt(v)); % d x T; 
dim = size(X,1);
p = -0.5 * (sum(normalized_diff * normalized_diff') + dim * log(2*pi) + sum(v));





function y_lf0 = get_lf0(lf0, lf0_gv, delta)
%T = size(lf0, 2); % [ (n+1)*d , T ]
%delta = [-0.5 0 0.5; 1 -2 1];

lf0_pre1 = lf0(4,:);
lf0_cur = lf0(5, :);
lf0_suc1 = lf0(6, :);

lf0_1 = -0.5*lf0_pre1 + 0.5 *lf0_suc1;
lf0_2 = lf0_pre1 - 2*lf0_cur + lf0_suc1;
lf0 = [lf0_cur; lf0_1; lf0_2];

mean_file = '/home/zhaoyi/jt/JT_Corpus_WangMiaoQi/data/mean';
gv_file = '/home/zhaoyi/jt/JT_Corpus_WangMiaoQi/data/gv';
dim_target = 121;
dim_vu = 1;
dim_lf0 = 1;
dim_mgc = (dim_target - dim_lf0*3 - dim_vu)/3;
lf0_range=[dim_mgc*3+1,dim_mgc*3+dim_lf0*3];

n_std=importdata([mean_file filesep 'output_std.mat']);

%lf0_gv_mu = gv_mu(lf0_range(1):lf0_range(1)+dim_lf0-1)';
%lf0_gv_var = gv_var(lf0_range(1):lf0_range(1)+dim_lf0-1)';

%lf0_gv=struct('mu',lf0_gv_mu,'Sigma',lf0_gv_var);
v = n_std.^2;

lf0_v = v(lf0_range(1):lf0_range(2))';

y_lf0 = mlpggv(lf0, lf0_v, delta, lf0_gv);








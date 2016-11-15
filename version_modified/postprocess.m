dim_vu = 1;
dim_lf0 = 9;
dim_mgc = (dim_target - dim_lf0*3 - dim_vu)/3;
mgc_range=[1,dim_mgc*3];
lf0_range=[dim_mgc*3+1,dim_mgc*3+dim_lf0*3];
delta = [-0.5 0 0.5; 1 -2 1];

mgc_dir = dir([gen_par_dyn filesep 'mgc' filesep '*.mgc']);
lf0_dir = dir([gen_par_dyn filesep 'lf0' filesep '*.lf0']);

mgc_out_dir = [gen_par filesep 'mgc'];
lf0_out_dir = [gen_par filesep 'lf0'];

if(~exist(mgc_out_dir, 'dir')) mkdir(mgc_out_dir); end
if(~exist(lf0_out_dir, 'dir')) mkdir(lf0_out_dir); end

len_mgc_files=length(mgc_dir);

n_std=importdata([mean_file filesep 'output_std.mat']);
gv_mu=importdata([gv_file filesep 'gv_mu.mat']);
gv_var=importdata([gv_file filesep 'gv_var.mat']);
gv_var=sqrt(gv_var);
mgc_gv_mu = gv_mu(mgc_range(1):mgc_range(1)+dim_mgc-1)';
lf0_gv_mu = gv_mu(lf0_range(1):lf0_range(1)+dim_lf0-1)';
mgc_gv_var = gv_var(mgc_range(1):mgc_range(1)+dim_mgc-1)';
lf0_gv_var = gv_var(lf0_range(1):lf0_range(1)+dim_lf0-1)';


mgc_gv=struct('mu',mgc_gv_mu,'Sigma',mgc_gv_var);
lf0_gv=struct('mu',lf0_gv_mu,'Sigma',lf0_gv_var);
v = n_std.^2;
mgc_v = v(mgc_range(1):mgc_range(2))';
lf0_v = v(lf0_range(1):lf0_range(2))';

for n=1:len_mgc_files
    basename=regexp(mgc_dir(n).name,'\.mgc','split');
    basename=char(basename(1));
    str=sprintf('Analysing file: %s',basename);

    fname_mgc = [gen_par_dyn filesep 'mgc' filesep basename '.mgc'];
    fname_lf0 = [gen_par_dyn filesep 'lf0' filesep basename '.lf0'];
    fname_vu = [gen_par_dyn filesep 'vu' filesep basename '.vu'];

    mgc_out = [mgc_out_dir filesep basename '.mgc'];
    lf0_out  = [lf0_out_dir filesep basename '.lf0'];

    fid = fopen (fname_mgc);
	mgc = fread (fid, [dim_mgc*3,inf],'float');
	fclose(fid);

	fid = fopen (fname_lf0);
	lf0 = fread (fid, [dim_lf0*3,inf],'float');
	fclose(fid);

	fid = fopen (fname_vu);
	vu = fread (fid,'float');
	fclose(fid);



	%%%%%%%%%%%%%% No postfiltering%%%%%%%%%%%%%%%%%%%%%%
	y_mgc = mgc(1:dim_mgc, :);
	y_lf0 = lf0(1:dim_lf0, :);
	%%%%%%%%%%%mlpgpostfiltering%%%%%%%%%%%%%%%%%%%%%%
	%y_mgc = mlpg(mgc, mgc_v, delta);
	%y_lf0 = mlpg(lf0, lf0_v, delta);
	
	%%%%%mlpg gv postfiltering %%%%%%%%%%%%%
	%y_mgc = mlpggv(mgc, mgc_v, delta, mgc_gv);
	%y_lf0 = mlpggv(lf0, lf0_v, delta, lf0_gv);




	y_lf0= y_lf0(5,:);

	for i = 1:length(y_lf0)
		if vu(i) < 0.6
	 		y_lf0(i) = -1e+10;
	 	end
	 end
	
	
	fid = fopen (mgc_out, 'w');
	fwrite(fid,y_mgc,'float');
	fclose(fid);

	fid=fopen(lf0_out,'w');
	fwrite(fid,y_lf0,'float');
	fclose(fid);

end
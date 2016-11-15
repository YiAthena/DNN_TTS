
targetlist=dir([file_target filesep '*.target']);
len_target_files=length(targetlist);

target_msum = [];
for n=1:len_target_files
	basename=regexp(targetlist(n).name,'\.target','split');
    basename=char(basename(1));
    str=sprintf('Reading file: %s',basename);
    disp(str)

    fid=fopen([file_target filesep basename '.target'],'r');
	data_target = fscanf(fid,'%f',[dim_target,inf])';
	fclose(fid);

	target_mean = mean(data_target,1);
	target_var = mean(data_target.^2,1)-target_mean.^2;
	target_msum = [target_msum; target_var];
end

gv_mu = mean(target_msum,1);
gv_var = mean(target_msum .^2,1) - (gv_mu).^2;
gv_std = sqrt(gv_var);

save([gv_path filesep 'gv_mu'],'gv_mu');
save([gv_path filesep 'gv_var'],'gv_var');
save([gv_path filesep 'gv_std'],'gv_std');
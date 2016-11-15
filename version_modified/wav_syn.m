mgc_dir = dir([gen_par filesep 'mgc' filesep '*.mgc']);
len_mgc_files=length(mgc_dir);
dim_vu = 1;
dim_lf0 = 9;
dim_mgc = (dim_target - dim_lf0*3 - dim_vu)/3;
mgc_range=[1,dim_mgc*3];
for n=1:len_mgc_files
    basename=regexp(mgc_dir(n).name,'\.mgc','split');
    basename=char(basename(1));
    str=sprintf('Analysing file: %s',basename);

    fname_mgc = [gen_par filesep 'mgc' filesep basename '.mgc'];
    fname_lf0 = [gen_par filesep 'lf0' filesep basename '.lf0'];

    fid = fopen (fname_mgc);
	mgc = fread (fid,[dim_mgc, inf],'float');
	fclose(fid);
	fid = fopen (fname_lf0);
	lf0 = fread (fid,'float');
	fclose(fid);
	f0 = exp(lf0);
	y_mgc=sinsynthesis_mgc(mgc,f0);
	y_mgc=0.5*y_mgc./max(abs(y_mgc));
    
    audiowrite([wav_path filesep basename '.wav'],y_mgc,16000,'BitsPerSample',32);
end



fileList=dir([cmp_dir filesep '*.cmp']);
N=length(fileList);
if N==0
    disp(['No cmp files in ',cmp_dir])
end
mkdir_if_not_exist(par_dir);

for n=1:N
    basename=regexp(fileList(n).name,'\.cmp','split');
    basename=char(basename(1));
    str=sprintf(' Analysing file: %s',basename);
    disp(str)
        
    	par=ReadCmp([cmp_dir filesep fileList(n).name]);
		
		lf0=par(:,mgc_dim*3+1);
    	lf01=par(:,mgc_dim*3+2);
    	lf02=par(:,mgc_dim*3+3);
		

		[lf0_in,lf01_in,lf02_in]=lf0_interp_linear(lf0);
       
		VU=get_VU(lf0);
		vec=[lf0_in lf01_in lf02_in VU];

        max_abs_in=max(abs(vec));

        if(max_abs_in>9999)
            str=sprintf('input data wrong in %s', find(vec,max_abs_in));
            disp(str)
            
        end

        dlmwrite([lf0_dir filesep basename '.target'],vec,'\t');

end
inputlist=dir([file_input filesep '*.csv']);
mean=importdata([mean_file filesep 'output_mean.mat']);
std=importdata([mean_file filesep 'output_std.mat']);
len_input_files=length(inputlist);

dim_vu = 1;
dim_lf0 = 9;
dim_mgc = (dim_target - dim_lf0*3 - dim_vu)/3;

mgc_range=[1,dim_mgc*3];
lf0_range=[dim_mgc*3+1,dim_mgc*3+dim_lf0*3];
vu_range=[dim_mgc*3+dim_lf0*3+1,dim_mgc*3+dim_lf0*3+dim_vu];

mgc_file=[file_output filesep 'mgc'];
lf0_file=[file_output filesep 'lf0'];
vu_file=[file_output filesep 'vu'];
mkdir(mgc_file);
mkdir(lf0_file);
mkdir(vu_file);

for n=1:len_input_files
        mgc=[];
        lf0=[];
        vu=[];
        data_output=[];
        basename=regexp(inputlist(n).name,'\.csv','split');
        basename=char(basename(1));
        str=sprintf('Analysing file: %s',basename);
        disp(str)
        data_input=csvread1([file_input filesep inputlist(n).name]);

        for j=1:size(data_input,1)
            data_output(j,:)=data_input(j,:).*(std)+mean;
        end
        mgc=data_output(:,mgc_range(1):mgc_range(2))';
		lf0=data_output(:,lf0_range(1):lf0_range(2))';
		vu=data_output(:,vu_range(1):vu_range(1))';

		mgc_name=[file_output filesep 'mgc' filesep basename '.mgc'];
		lf0_name=[file_output filesep 'lf0' filesep basename '.lf0'];
		vu_name=[file_output filesep 'vu' filesep basename '.vu'];


		fid=fopen(mgc_name,'w');
		fwrite(fid,mgc,'float');
		fclose(fid);
		fid=fopen(lf0_name,'w');
		fwrite(fid,lf0,'float');
		fclose(fid);
		fid=fopen(vu_name,'w');
		fwrite(fid,vu,'float');
		fclose(fid);
		
end
exit;
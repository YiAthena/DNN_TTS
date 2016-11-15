inputlist=dir([file_input filesep '*.input']);
targetlist=dir([file_target filesep '*.target']);
len_input_files=length(inputlist);
len_target_files=length(targetlist);
assert(len_input_files == len_target_files, 'the number of input_files %d is not equal to the number of target_files %d.', len_input_files, len_target_files);
numSeqs = len_input_files;
numTimesteps = 0;
inputPattSize = 0;
targetPattSize = 0;
maxSeqTagLength = 0;

input_msum = [];
output_msum = [];

for n=1:len_input_files
    %data_input=[];
    data_target=[];
    basename=regexp(inputlist(n).name,'\.input','split');
    basename=char(basename(1));
    str=sprintf('Reading file: %s',basename);
    disp(str)

	%fid=fopen([file_input filesep basename '.input'],'r');
	%data_input = fscanf(fid,'%d',[dim_input,inf])';
	%fclose(fid);
	

	fid=fopen([file_target filesep basename '.target'],'r');
	data_target = fscanf(fid,'%f',[dim_target,inf])';
	fclose(fid);

	%assert(size(data_input,1)==size(data_target,1),'frame error');
 
    %maxSeqTagLength = max(maxSeqTagLength,length(basename));
    %numTimesteps = numTimesteps + size(data_input,1);
    %inputPattSize = size(data_input,2);
    %targetPattSize = size(data_target,2);

    % input_msum = [input_msum;data_input];
     output_msum = [output_msum;data_target];
end


 %input_mean = mean(input_msum,1);
 % input_std = std(input_msum,0,1);
 % input_std(input_std==0)=0.01;
  output_mean = mean(output_msum,1);
  output_std = std(output_msum,0,1);
  output_std(output_std==0)=0.01;
 % save([mean_dir filesep 'input_mean'],'input_mean');
 % save([mean_dir filesep 'input_std'],'input_std');
  save([mean_dir filesep 'output_mean'],'output_mean');
  save([mean_dir filesep 'output_std'],'output_std');
  exit;
input_mean=importdata([mean_dir filesep 'input_mean.mat']);
input_std=importdata([mean_dir filesep 'input_std.mat']);
output_mean=importdata([mean_dir filesep 'output_mean.mat']);
output_std=importdata([mean_dir filesep 'output_std.mat']);

   
ncid  = netcdf.create(nc_filename ,'NETCDF4');

numSeqsId  = netcdf.defDim(ncid ,'numSeqs',numSeqs );
numTimestepsId  = netcdf.defDim(ncid ,'numTimesteps',numTimesteps );
inputPattSizeId  = netcdf.defDim(ncid ,'inputPattSize',inputPattSize );
maxSeqTagLengthId  = netcdf.defDim(ncid ,'maxSeqTagLength',maxSeqTagLength );
targetPattSizeId  = netcdf.defDim(ncid ,'targetPattSize',targetPattSize );

seqTagsID  = netcdf.defVar(ncid ,'seqTags','char',[maxSeqTagLengthId  numSeqsId ]);
seqLengthsID  = netcdf.defVar(ncid ,'seqLengths','int',numSeqsId );
inputsID  = netcdf.defVar(ncid ,'inputs','float',[inputPattSizeId  numTimestepsId ]);
targetPatternsID  = netcdf.defVar(ncid ,'targetPatterns','float',[targetPattSizeId  numTimestepsId ]);
netcdf.endDef(ncid );

frameIndex = 0;
fileIndex = 0;
for n=1 : len_input_files
    data_input=[];
    data_target=[];

    basename=regexp(inputlist(n).name,'\.input','split');
    basename=char(basename(1));
    str=sprintf('Writing file: %s',basename);
    disp(str)
    
    fid=fopen([file_input filesep basename '.input'],'r');
	data_input = fscanf(fid,'%d',[dim_input,inf])';
	fclose(fid);
	

	fid=fopen([file_target filesep basename '.target'],'r');
	data_target = fscanf(fid,'%f',[dim_target,inf])';
	fclose(fid);

    for j=1:size(data_input,1)
        data_input(j,:) = (data_input(j,:)-input_mean)./(input_std);
        data_target(j,:) = (data_target(j,:)-output_mean)./(output_std);
    end
    
    netcdf.putVar(ncid ,inputsID ,[0 frameIndex],[size(data_input,2) size(data_input,1)],data_input');
    netcdf.putVar(ncid ,targetPatternsID ,[0 frameIndex],[size(data_target,2) size(data_target,1)],data_target');

    netcdf.putVar(ncid ,seqTagsID ,[0 fileIndex],[length(basename) 1],basename);
    netcdf.putVar(ncid ,seqLengthsID ,fileIndex,1,size(data_target,1));

    fileIndex = fileIndex + 1;
    frameIndex = frameIndex + size(data_target,1);
end
netcdf.close(ncid);


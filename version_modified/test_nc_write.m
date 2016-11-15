inputlist=dir([file_input filesep '*.input']);
len_input_files=length(inputlist);

numSeqs = len_input_files;
numTimesteps = 0;
inputPattSize = 0;
targetPattSize = 0;
maxSeqTagLength = 0;

for n=1:len_input_files
        data_input=[];
        basename=regexp(inputlist(n).name,'\.input','split');
        basename=char(basename(1));
        str=sprintf('Reading file: %s',basename);
        disp(str)

        fid=fopen([file_input filesep basename '.input'],'r');
		data_input = fscanf(fid,'%d',[dim_input,inf])';
		fclose(fid);
	maxSeqTagLength = max(maxSeqTagLength,length(basename));
    numTimesteps = numTimesteps + size(data_input,1);
    
end

inputPattSize = dim_input;
targetPattSize = dim_target;

input_mean=importdata([mean_dir filesep 'input_mean.mat']);
input_std=importdata([mean_dir filesep 'input_std.mat']);
ncid  = netcdf.create(nc_filename ,'CLASSIC_MODEL');

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

    for j=1:size(data_input,1)
        data_input(j,:) = (data_input(j,:)-input_mean)./(input_std);
        data_target(j,:) = zeros(1, dim_target);
    end
    
    netcdf.putVar(ncid ,inputsID ,[0 frameIndex],[size(data_input,2) size(data_input,1)],data_input');
    netcdf.putVar(ncid ,targetPatternsID ,[0 frameIndex],[size(data_target,2) size(data_target,1)],data_target');

    netcdf.putVar(ncid ,seqTagsID ,[0 fileIndex],[length(basename) 1],basename);
    netcdf.putVar(ncid ,seqLengthsID ,fileIndex,1,size(data_target,1));

    fileIndex = fileIndex + 1;
    frameIndex = frameIndex + size(data_target,1);
end
netcdf.close(ncid);

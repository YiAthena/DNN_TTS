
file_input = '/home/zhaoyi/jt/JT_Corpus_WangMiaoQi/data4/INPUT';
inputlist=dir([file_input filesep '*.input']);

len_input_files=length(inputlist);

dim_input = 192;

file_mean = [];
file_var = [];
file_power_mean = [];
file_frame =[];

for n=1:len_input_files

    data_input=[];
    basename=regexp(inputlist(n).name,'\.input','split');
    basename=char(basename(1));
    str=sprintf('Reading file: %s',basename);
    disp(str)

    fid=fopen([file_input filesep basename '.input'],'r');
    data_input = fscanf(fid,'%d',[dim_input,inf])';
    fclose(fid);

    file_mean =[file_mean; mean(data_input, 1)]; 
    file_var = [file_var; var(data_input, 1)];
    file_power_mean = [file_power_mean; mean(data_input, 1).^2 + var(data_input, 1)];
    file_frame = [file_frame; size(data_input, 1)];

end

for i = 1:length(file_frame)
    file_mean(i, :) = file_mean(i, :) .* file_frame(i);
    file_power_mean(i, :) = file_power_mean(i, :) .* file_frame(i);
end
input_mean=sum(file_mean, 1)/sum(file_frame);
input_power_mean = sum(file_power_mean, 1)/sum(file_frame);
input_var = input_power_mean - input_mean.^2;
input_std = sqrt(input_var);    
input_std(input_std == 0) = 0.01;

    



   


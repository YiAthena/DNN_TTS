train_input=/home/zhaoyi/jt/JT_Corpus_WangMiaoQi/data6/INPUT
train_target=/home/zhaoyi/jt/JT_Corpus_WangMiaoQi/data6/TARGET
mean_dir=/home/zhaoyi/jt/JT_Corpus_WangMiaoQi/data6/mean
nc_file=/home/zhaoyi/jt/JT_Corpus_WangMiaoQi/data6/nc/train.nc
switch=train
dim_input=192
dim_target=129
matlab -nojvm -nosplash -nodesktop -r "file_input='$train_input', file_target='$train_target', 	mean_dir='$mean_dir', nc_filename='$nc_file', dim_input=$dim_input,dim_target=$dim_target; 	train_nc_write;quit"

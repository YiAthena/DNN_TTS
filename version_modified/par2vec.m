%function par2vec(cmp_dir, par_dir, mgc_dim) %input dir output dir mgc_dim

%cmp_dir = cmp_dir;
%par_dir = par_dir;
%mgc_dim = mgc_dim;
fileList=dir([cmp_dir filesep '*.cmp']);
N=length(fileList);
delta = [-0.5 0 0.5; 1 -2 1];
if N==0
    disp(['No cmp files in ',cmp_dir])
end
mkdir_if_not_exist(par_dir);

for n=1:N
    basename=regexp(fileList(n).name,'\.cmp','split');
    basename=char(basename(1));
    str=sprintf(' Analysing file: %s',basename);
    disp(str)
    
    %try
        
    	par=ReadCmp([cmp_dir filesep fileList(n).name]);
		mgc=par(:,1:mgc_dim);
		mgc1=par(:,(mgc_dim+1):(mgc_dim*2));
		mgc2=par(:,(mgc_dim*2+1):(mgc_dim*3));
		lf0=par(:,mgc_dim*3+1);
    	lf01=par(:,mgc_dim*3+2);
    	lf02=par(:,mgc_dim*3+3);
		

		[lf0_in,lf01_in,lf02_in]=lf0_interp_linear(lf0);
        
       
		VU=get_VU(lf0);
		
        lf0_pre1 = [lf0_in(1); lf0_in(1:size(lf0_in, 1)-1)];
        lf0_pre2 = [lf0_pre1(1); lf0_pre1(1:size(lf0_pre1, 1)-1)];
        lf0_pre3 = [lf0_pre2(1); lf0_pre2(1:size(lf0_pre1, 1)-1)];
        lf0_pre4 = [lf0_pre3(1); lf0_pre3(1:size(lf0_pre1, 1)-1)];

        lf0_suc1 = [lf0_in(2:size(lf0_in, 1)); lf0_in(size(lf0_in, 1))];
        lf0_suc2 = [lf0_suc1(2:size(lf0_suc1, 1)); lf0_suc1(size(lf0_suc1, 1))];
        lf0_suc3 = [lf0_suc2(2:size(lf0_suc2, 1)); lf0_suc2(size(lf0_suc2, 1))];
        lf0_suc4 = [lf0_suc3(2:size(lf0_suc3, 1)); lf0_suc3(size(lf0_suc3, 1))];

        lf0_mat = [lf0_pre4 lf0_pre3 lf0_pre2 lf0_pre1 lf0_in lf0_suc1 lf0_suc2 lf0_suc3 lf0_suc4 lf01_in lf02_in];
        %lf0_mat = lf0_mat';
        %lf0_delta_mat = addDelta(lf0_mat, delta)';
        %vec=[mgc mgc1 mgc2 lf0_delta_mat VU];
        

        vec=[mgc mgc1 mgc2 lf0_mat VU];


        max_abs_in=max(abs(vec));

        if(max_abs_in>9999)
            str=sprintf('input data wrong in %s', find(vec,max_abs_in));
            disp(str)
            
        end

        dlmwrite([par_dir filesep basename '.target'],vec,'\t');

    %catch 
    %    str=sprintf('.............ERROR NOT ANALYSED!!!');
    %    disp(str)
    %end
end


% function VU=get_VU(lf0)
%     VU=zeros(length(lf0),1);
%     f0=exp(lf0);
% 	for i=1:length(lf0)
% 		if f0(i)>50
% 			VU(i)=1;
        
%         else
% 			VU(i)=0;
%         end
%     end
% end



% function [lf0_in,lf01_in,lf02_in]=lf0_interp_linear(lf0)
%     f0=exp(lf0);
%     x=find(f0);
%     y=f0(x);
%     if(1~=x(1))
%    		x=[1;x];
%         y=[y(1);y];
%     end
%     if(x(length(x))~=length(f0))
%        x=[x;length(f0)];
%    	   y=[y;y(length(y))];
%     end
    
%     xi=(1:length(lf0))';
%     f0_in=interp1(x,y,xi,'linear');
% 	lf0_in=log(f0_in);
  
%     lf01_in=zeros(length(lf0),1);
%     lf02_in=zeros(length(lf0),1);
%     for i=2:(length(lf0_in)-1)
%         lf01_in(i)=0.5*(lf0_in(i+1)-lf0_in(i-1));
%         lf02_in(i)=lf0_in(i+1)-2*lf0_in(i)+lf0_in(i-1);
%     end
% end

%  function [lf0_in,lf01_in,lf02_in]=lf0_interp_spline(lf0)
%     f0=exp(lf0);
%     x=find(f0);
%     y=f0(x);
%     x=[1;x;length(f0)];
%     y=[0;y;0];
%     xi=(1:length(f0))';
    
%     f0_in=interp1(x,y,xi,'spline');
%     lf0_in=log(f0_in);
%     lf01_in=zeros(length(lf0),1);
%     lf02_in=zeros(length(lf0),1);
%     for i=2:(length(lf0_in)-1)
%         lf01_in(i)=0.5*(lf0_in(i+1)-lf0_in(i-1));
%         lf02_in(i)=lf0_in(i+1)-2*lf0_in(i)+lf0_in(i-1);
%     end
% end  



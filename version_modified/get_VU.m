function VU=get_VU(lf0)
    VU=zeros(length(lf0),1);
    f0=exp(lf0);

	for i=1:length(lf0)
		if f0(i)>50
			VU(i)=1;
        
        else
			VU(i)=0;
        end
    end
end
function [lf0_in,lf01_in,lf02_in]=lf0_interp_linear(lf0)
    f0=exp(lf0);
    x=find(f0);
    y=f0(x);
    if(1~=x(1))
   		x=[1;x];
        y=[y(1);y];
    end
    if(x(length(x))~=length(f0))
       x=[x;length(f0)];
   	   y=[y;y(length(y))];
    end
    
    xi=(1:length(lf0))';
    f0_in=interp1(x,y,xi,'linear');
	lf0_in=log(f0_in);
  
    lf01_in=zeros(length(lf0),1);
    lf02_in=zeros(length(lf0),1);
    for i=2:(length(lf0_in)-1)
        lf01_in(i)=0.5*(lf0_in(i+1)-lf0_in(i-1));
        lf02_in(i)=lf0_in(i+1)-2*lf0_in(i)+lf0_in(i-1);
    end
end
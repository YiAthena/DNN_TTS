function y=sinsynthesis_mgc(mgc,f0)
ww=triang(160);
[m,n]=size(mgc);
y=zeros(n*80+80,1);
pf0=81;
qz=0;
for j=1:n
   fc=f0(j);
   if fc<=10
       fc=100;
       qz=1;
   else
       qz=0;
   end
   fnum=floor(7800/fc);
   har=zeros(fnum,1);
   ang=zeros(fnum,1);
   for t=1:fnum
       fw=freqwrap(2*pi*t*fc/16000);
       for k=1:m
           har(t)=har(t)+mgc(k,j)*cos(fw*(k-1));
           ang(t)=ang(t)-mgc(k,j)*sin(fw*(k-1));
       end
   end
   ms=max(exp(har));
   har=exp(har);%.*((1:length(har))'+5);
   %har=har/max(har)*ms;
   frame=zeros(160,1);
   if fc>0
       kk=fnum;
       if j>1
           if f0(j-1)>0
               pf0=pf0-80;
               while abs(pf0 +32000/(f0(j-1)+fc) -80 )< abs(pf0-80) %pf0+32000/(f0(j-1)+fc)<80 %
                   pf0=pf0+32000/(f0(j-1)+fc);
               end
           end
       end
       for t=1:kk 
           sc=har(t);
           ar=ang(t);
           u=randn(1)*10*pi;
           for k=1:160
               if t*fc<4000 && qz==0
                   frame(k)=frame(k)+ abs(sc)*cos((k-pf0)*2*pi*t*fc/16000+ar);
               else
                   frame(k)=frame(k)+ abs(sc)*cos((k-pf0)*2*pi*t*fc/16000+u);
               end
           end
       end
       y( (j-1)*80+1: j*80+80)= y((j-1)*80+1:j*80+80)+ frame.*ww/sqrt(kk);
   end
end
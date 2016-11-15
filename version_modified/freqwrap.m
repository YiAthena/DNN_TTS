function y=freqwrap(x);
alpha=0.42;
a=(1-alpha*alpha)*sin(x);
b=(1+alpha*alpha)*cos(x)-2*alpha;
if b==0
    y=pi/2;
else
    y=atan(a/b);
    if y<0
        y=y+pi;
    end
end
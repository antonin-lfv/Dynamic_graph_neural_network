function [Xmat,Tmat]=syllablecut(data,fs,Filtl,Filts,minsp,maxsp);

% data : data sequence, (N x 1) samples
% fs: sampling frequency
% Filtl: Long smoothing filter in msec
% Filts: Short smoothing filter in msec
% minsp: minimum allowed time space between two syllables in msec, default 100 msec.
% maxsp: maximum allowed syllable length in msec, default 2000 msec
%
% Xmat: Matrix with detected strophes/syllables as column vectors
% Tmat: Matrix with corresponding time intervals in sec.

%figoff=1; %No figures show
figoff=0; %Figures show


if nargin<6
    maxsp=2000;
end
if nargin<5
    minsp=100;
end

extth=Filts;

maxt=max(find(data~=0))
data=data(1:maxt);


Filts=fix(Filts*fs/1000); %
Filtl=fix(Filtl*fs/1000);


Next=fix(extth*fs/1000); 
bets=minsp*fs/1000; 
%aets=maxsp*fs/1000; 

xpows=conv(ones(Filts,1)/Filts,data.^2);
xpows=xpows(Filts/2+1:length(data)+Filts/2);
xpowl=conv(ones(Filtl,1)/Filtl,data.^2);
xpowl=xpowl(Filtl/2+1:length(data)+Filtl/2);
t=[0:length(data)-1];


if figoff==0
figure
subplot(211)

plot(t/fs,[xpows xpowl],'LineWidth',1.2)
title('a) The two smoothing power filters, and detected samples above threshold (red)') 
ylabel('Amplitude^2')
xlabel('s')
hold on

end

lev=1/100*max(xpowl); %Level above threshold for detected samples 


s=zeros(length(data),1);
for i=1:length(data)
 if (xpows(i)>xpowl(i)+lev);
    s(i)=1;
 end
end
ss=find(s==1);

if figoff==0
plot(ss/fs,xpows(ss),'r.')
xlabel('s')


hold off

subplot(212)

plot(t/fs,real(data))
xlabel('s')
title('Signal')
ylabel('Amplitude')
hold on
end


ss=[1;ss;length(data)];
sub=find(diff(ss)>bets);



%for i=1:length(sub)-1
%  if sub(i+1)-sub(i)>aets
%    subadd=find(diff(ss([sub(i)+1:sub(i+1)-1]))>bets/2)+sub(i);
%    sub=[sub;subadd];
%
%
%  end
%end
sub=sort(sub);

sylllim=0.1*max(max(abs(data)));
  
  
for i=1:length(diff(sub))
    in=[max(1,ss(sub(i)+1)-Next):min(length(data),ss(sub(i+1))+Next)];
    
    if figoff==0
    plot([min(in) max(in)]/fs,(-sylllim+sylllim/4*(-1)^i)*[1 1],'m X')
    plot([min(in) max(in)]/fs,(-sylllim+sylllim/4*(-1)^i)*[1 1],'m -')
    %text(min(in)/fs-0.01,max(max(real(data))),int2str(i))
    axis([0 max(t)/fs min(data)*1.2 max(data)*1.2])
    %title('b) Signal with detected strophes/syllables') 
    end
    xx=data(in);
    tt=in;
    Xmat(1:length(xx),i)=xx;
    Tmat(1:length(tt),i)=tt/fs;
end


%if figoff==0
%hold off
%pause(0.5)
%sound(data0,44100)
%pause
%end
  

  
  
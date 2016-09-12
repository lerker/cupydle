function [F0_mean,F0_std,energyST_mean,energyST_std] = F0_energy(s,fs)

% Estimacion del pitch, con ventanas de 10 ms
[fx,~,pv]= fxpefac(s,fs,0.01); 

% Cortamos la parte sin voz de la señal en el tiempo
% para calcular la energia solo de la parte de voz

msk=pv>0.5; % find voiced frames as a mask
fxg=fx;
fxg(~msk)=[];




pp.pr=0.99; % definimos umbral de voz
pp.ts=800; % media en ms de rafaga de voz
pp.tn=15; % media en ms silencios
pp.ne=1; % estamacion ruido MMSE
pp.ta=0.9; % cte de tiempo para suavizar SNR 

%Recortamos el 68% del primer segundo y el 79% del ultimo segundo del video
%para eliminar ruido de grabacion
fi_recorte = round(fs*0.6802);
ff_recorte = round(fs*0.7936);

mskt=vadsohn(s(fi_recorte:end-ff_recorte),fs,'a',pp); % detector de voz



sg=s(fi_recorte:end-ff_recorte);
sg=sg.*mskt; % aplicamos máscara
mskt2=abs(sg)>0; 
sg(~mskt2)=[]; % cortamos donde no hay voz

%subplot(2,1,1)
%plot(s)
%subplot(2,1,2)
%plot(sg)

% Short time energy
tv=fs*0.01;
winLen = tv; % ventana de 10 ms 44100*0.01
winOverlap = tv-1;
wHamm = hamming(winLen);

% Framing and windowing the signal 
sgFramed = buffer(sg, winLen, winOverlap, 'nodelay');

sgWindowed = diag(sparse(wHamm))*sgFramed;

% Short-Time Energy calculation
energyST = sum(sgWindowed.^2,1);

% Time in seconds, for the graphs
%t = [0:length(sg)-1]/fs;

%subplot(1,1,1);
%plot(t, sg);
%title('speech: miyamufada');
%xlims = get(gca,'Xlim');

%hold on;

% Short-Time energy is delayed due to lowpass filtering. This delay is
% compensated for the graph.
%delay = (winLen - 1)/2;
%plot(t(delay+1:end - delay), energyST, 'r');
%xlim(xlims);
%xlabel('Time (sec)');
%legend({'Speech','Short-Time Energy'});
%hold off;

F0_mean=mean(fxg);
F0_std=std(fxg);
energyST_mean=mean(energyST);
energyST_std=std(energyST);

end


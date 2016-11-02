function [ ] = voice_t0(signalpath)
%Retorno en frame en video ini y fin cuando la voz en la pista de audio
%comienza

[s,fs]=audioread(signalpath);

pp.pr=0.99; % definimos umbral de voz
pp.ts=800; % media en ms de rafaga de voz
pp.tn=15; % media en ms silencios
pp.ne=1; % estimacion ruido MMSE
pp.ta=0.9; % cte de tiempo para suavizar SNR 

%Recortamos el 68% del primer segundo y el 79% del ultimo segundo del video
%para eliminar ruido de grabacion
fi_recorte = round(fs*0.6802);
ff_recorte = round(fs*0.7936);

msk=vadsohn(s(fi_recorte:end-ff_recorte),fs,'a',pp); % detector de voz

%volvemos a sumar los frames q habiamos eliminado
ini=find(msk,1)+fi_recorte;
fin=find(msk,1,'last')+fi_recorte;

t_i=ini/fs; %tiempo inicial de voz
t_f=fin/fs; %tiempo final de voz

f_i=floor(t_i*30); %frame inicial de video
f_f=floor(t_f*30); %frame final de video

foutput = strrep(signalpath,'.wav','-voiced_frames.txt');
csvwrite(foutput,[f_i f_f]);

end


function [ mls_coef ] = MLS(s,sr)

% Entradas:
% s -> señal en el dominio temporal
% n -> cantidad MLS = 30 para el rango (0 - 1200 hz)
% sr -> frecuencia de muestreo
%Salidas:
% mls_coef -> vector MLS con n objetos

n=30;
mls_coef = zeros(1,n);

%Defino fft de s en escala log
S = fft(s);
%Calculo magnitud de S en escala log
Smag = abs(S);
Smag = log(Smag);
N = length(S);


df=sr/N;
f=[-sr/2:df:sr/2-df];

%---------------------------------------------------------
% Cortamos la parte que no hay voz de la señal en el tiempo 
pp.pr=0.99; % definimos umbral de voz
pp.ts=800; % media en ms de rafaga de voz
pp.tn=15; % media en ms silencios
pp.ne=1; % estamacion ruido MMSE
pp.ta=0.9; % cte de tiempo para suavizar SNR 

%Recortamos el 68% del primer segundo y el 79% del ultimo segundo del video
%para eliminar ruido de grabacion
fi_recorte = round(sr*0.6802);
ff_recorte = round(sr*0.7936);

mskt=vadsohn(s(fi_recorte:end-ff_recorte),sr,'a',pp); % detector de voz

sg=s(fi_recorte:end-ff_recorte); % 
sg=sg.*mskt; % aplicamos máscara
mskt2=abs(sg)>0; 
sg(~mskt2)=[]; % cortamos donde no hay voz

Sg = fft(sg); % transformada de la señal cortada
SmagG = abs(Sg);
SmagG = log(SmagG); % espectro logaritmico

Ng = length(Sg);

dfg= sr/Ng; % resolucion frecuencial
fg=[-sr/2:dfg:sr/2-dfg]; % para graficar

% subplot(2,1,1);
% stem(f,fftshift(Smag));
% subplot(2,1,2);
% stem(fg,fftshift(SmagG));


%---------------------------------------------------------

%Defino df y calculo cuantas muestras entran en cada uno de los 30 bloques
Ninteres = 1200/dfg;
cNxbloque = floor(Ninteres/n);

%Recorro la transformada y acumulo por banda
sum=0;
k=0;
c=1;
while(c<n+1)
    for j=1:cNxbloque
        sum=sum+SmagG(j+k);
    end
    k=k+cNxbloque;
    mls_coef(c)=sum/cNxbloque;
    sum=0;
    c=c+1;
end



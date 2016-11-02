function [  ] = features(fsignal)
    %FEATURES Summary of this function goes here
    %   Detailed explanation goes here

    [s,fs] = audioread(fsignal);

    % 12 Mel coeffs y sus derivadas primera y segunda

    % Por defecto: ventana: 1024 frames = 23ms [potencia de 2 < (0.03*fs)]
    % Por defecto: salto 512 frames (11.5 ms)
    % Por defecto: Hamming window in time domain
	% melcepst(s,fs,'dD') -> con primeras y segundas derivadas
    c = mean(melcepst(s,fs));

    % 30 MLS (Mean Log-Spectrum)

    mls = MLS(s,fs);

    % Promedio y desviacion estandar de F0 y de la energia de tiempo corto

    [f0_mean,f0_std,energyST_mean,energyST_std]=F0_energy(s,fs);
    
    salida=[c mls f0_mean f0_std energyST_mean energyST_std];
    
    
    foutput = strrep(fsignal,'.wav','-audio_features.csv');
    dlmwrite(foutput,salida,'delimiter',' ','precision','%.6f');


end


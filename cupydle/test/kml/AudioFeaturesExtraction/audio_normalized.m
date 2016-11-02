function [Y] = audio_normalized(path)
% Normalizacion entre -1 y 1 de los datos de audio

X=dlmread(path,' ');

col = size(X,2);

for i=1:col
    Y(:,i) = X(:,i)/max(abs((X(:,i))));    
end

%Escribo archivo
foutput = strrep(path,'.csv','-normalized.csv');
    dlmwrite(foutput,Y,'delimiter',' ','precision','%.6f');

end


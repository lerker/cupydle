
%Recibe una ruta con archivos .wav y retorna para cada uno un archivo .csv
%con las caract. de audio de features.m
myFiles = dir(fullfile('wavs','*.wav'));
for k = 1:length(myFiles)
    length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile('wavs', baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    features(fullFileName);
end


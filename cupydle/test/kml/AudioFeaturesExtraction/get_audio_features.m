function [] = get_audio_features(myDir)
%Recibe una ruta con archivos .wav y retorna para cada uno un archivo .csv
%con las caract. de audio de features.m
myFiles = dir(fullfile(myDir,'*.wav')); %gets all wav files in struct
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
    
  features(fullFileName);
end

end

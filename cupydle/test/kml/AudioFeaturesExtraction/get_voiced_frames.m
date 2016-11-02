function [] = get_voiced_frames(myDir)
%Recibe una ruta con archivos .wav y retorna para cada uno un archivo .txt
% con el frame inicial y final en video donde hay voz.
myFiles = dir(fullfile(myDir,'*.wav')); %gets all wav files in struct
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
    
  voice_t0(fullFileName); %Frames inicial y final de video

end

end


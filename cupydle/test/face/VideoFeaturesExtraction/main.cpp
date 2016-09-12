#include "functions.h"

int main(int argc, char **argv)
{
    //if (argc!=2)
      //  std::cout<<"Uso main archivo.wav";

     cv::Mat boca;
     cv::Mat ojos;
     extrae_caract(argv[1],boca,ojos);
     std::string str=argv[1];
     int position = str.find ('.');
     str.replace(position,4,"-video_features.csv");

     writeMatsToFile(boca,ojos,str.c_str());     

return 0;
}

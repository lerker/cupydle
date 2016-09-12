#include "flandmark_detector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#endif // FUNCTIONS_H

IplImage* getCameraFrame(CvCapture* &camera, const char *filename , int camid, int width, int height);
void detectFaceInImage(IplImage *orig, IplImage* input, CvHaarClassifierCascade* cascade, FLANDMARK_Model *model, int *bbox, double *landmarks);
cv::Mat asRowMatrix(const std::vector<cv::Mat>& src, int rtype, double alpha, double beta);
void writeMatsToFile(cv::Mat& m, cv::Mat& m2, const char* filename);
void extrae_caract(char* path,cv::Mat & datos_boca,cv::Mat & datos_ojos);
void region_boca(IplImage* input,double *landmarks,cv::Mat & output);
void region_ojos(IplImage* input,double *landmarks,cv::Mat & output);







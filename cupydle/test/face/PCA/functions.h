#include "flandmark_detector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#endif // FUNCTIONS_H
using namespace std;

void perform_PCA(char *path);
void parseCSV(char *path, vector<vector<float> > & patterns );

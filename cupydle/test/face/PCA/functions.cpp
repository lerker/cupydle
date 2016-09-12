#include "functions.h"
#include <math.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

void writeMatsToFile(cv::Mat& m, const char* filename)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols-1; j++)
        {
            fout<<m.at<float>(i,j)<<" ";
        }
        fout<<m.at<float>(i,m.cols-1)<<"\n";
    }

    fout.close();
}

void parseCSV(char *path, vector<vector<float> > &patterns ){
    ifstream file(path);
    string line;
    while (getline(file,line))
    {   vector<float> v_aux;
    istringstream linestream(line);
    string data;
    while (getline(linestream, data, ' ')){
        double a= atoi(data.c_str());
        v_aux.push_back(a);
    }
    patterns.push_back(v_aux);
    }
    file.close();
}
// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

void perform_PCA(char *path){

    //Cargo datos desde archivo all_videos
    vector<vector<float> > data;
    parseCSV(path,data);


    //Cargo datos Anger
    vector<vector<float> >::const_iterator first_an = data.begin();
    vector<vector<float > >::const_iterator last_an = data.begin() + 120;
    vector<vector<float > > anger(first_an, last_an);
    //Cargo datos Disgust
    vector<vector<float> >::const_iterator first_di = data.begin()+ 120;
    vector<vector<float > >::const_iterator last_di = data.begin() + 240;
    vector<vector<float > > disgust(first_di, last_di);
    //Cargo datos Fear
    vector<vector<float> >::const_iterator first_fe = data.begin()+ 240;
    vector<vector<float > >::const_iterator last_fe = data.begin() + 360;
    vector<vector<float > > fear(first_fe, last_fe);
    //Cargo datos Happy
    vector<vector<float> >::const_iterator first_ha = data.begin()+ 360;
    vector<vector<float > >::const_iterator last_ha = data.begin() + 480;
    vector<vector<float > > happy(first_ha, last_ha);
    //Cargo datos sadness
    vector<vector<float> >::const_iterator first_sa = data.begin()+ 480;
    vector<vector<float > >::const_iterator last_sa = data.begin() + 600;
    vector<vector<float > > sad(first_sa, last_sa);
    //Cargo datos surprise
    vector<vector<float> >::const_iterator first_su = data.begin()+ 600;
    vector<vector<float > >::const_iterator last_su = data.begin() + 720;
    vector<vector<float > > sur(first_su, last_su);


    vector<Mat> pca_points; //Vector para almacenar proyecciones en cada nuevo espacio
    int num_components = 70; //Numero de componentes a retener en PCA
    //double var=0.85;


    //Creo una Mat para poder aplicar PCA Anger
    cv::Mat matanger(anger.size(), anger.at(0).size(), CV_32FC1);
    for(int i=0; i<matanger.rows; ++i)
         for(int j=0; j<matanger.cols; ++j)
              matanger.at<float>(i, j) = anger.at(i).at(j);


    //PCA sobre datos Anger
    PCA pca_an(matanger, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //Quiero saber la Varianza total retenida
    cout<<"numero de componentes anger: "<<pca_an.eigenvalues.rows<<endl;


    //Proyecta todos las imagenes en el nuevo espacio
    for(int i=0;i<matanger.rows;i++){
       Mat point,dst;
       point = pca_an.project(matanger.row(i));
       cv::normalize(point, dst, -1, 1, NORM_MINMAX,CV_32FC1); //Normalizacion entre -1 y 1

       pca_points.push_back(dst);
    }
    //---------------------------------------------------------------------------------------

    //Creo una Mat para poder aplicar PCA Disgust
    cv::Mat matdisgust(disgust.size(), disgust.at(0).size(), CV_32FC1);
    for(int i=0; i<matdisgust.rows; ++i)
         for(int j=0; j<matdisgust.cols; ++j)
              matdisgust.at<float>(i, j) = disgust.at(i).at(j);


    //PCA sobre datos Disgust
    PCA pca_di(matdisgust, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //Quiero saber la Varianza total retenida
    cout<<"numero de componentes disgust: "<<pca_di.eigenvalues.rows<<endl;

    //Proyecta todos las imagenes en el nuevo espacio
    for(int i=0;i<matdisgust.rows;i++){
       Mat point,dst;
       point = pca_di.project(matdisgust.row(i));
       cv::normalize(point, dst, -1, 1, NORM_MINMAX,CV_32FC1); //Normalizacion entre -1 y 1
       pca_points.push_back(dst);
    }
    //---------------------------------------------------------------------------------------

    //Creo una Mat para poder aplicar PCA Fear
    cv::Mat matfear(fear.size(), fear.at(0).size(), CV_32FC1);
    for(int i=0; i<matfear.rows; ++i)
         for(int j=0; j<matfear.cols; ++j)
              matfear.at<float>(i, j) = fear.at(i).at(j);


    //PCA sobre datos Fear
    PCA pca_fe(matfear, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //Quiero saber la Varianza total retenida
    cout<<"numero de componentes fear: "<<pca_fe.eigenvalues.rows<<endl;

    //Proyecta todos las imagenes en el nuevo espacio
    for(int i=0;i<matfear.rows;i++){
       Mat point,dst;
       point = pca_fe.project(matfear.row(i));
       cv::normalize(point, dst, -1, 1, NORM_MINMAX,CV_32FC1); //Normalizacion entre -1 y 1
       pca_points.push_back(dst);
    }
    //---------------------------------------------------------------------------------------

    //Creo una Mat para poder aplicar PCA Happy
    cv::Mat mathappy(happy.size(), happy.at(0).size(), CV_32FC1);
    for(int i=0; i<mathappy.rows; ++i)
         for(int j=0; j<mathappy.cols; ++j)
              mathappy.at<float>(i, j) = happy.at(i).at(j);


    //PCA sobre datos Happy
    PCA pca_ha(mathappy, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //Quiero saber la Varianza total retenida
    cout<<"numero de componentes happy: "<<pca_ha.eigenvalues.rows<<endl;

    //Proyecta todos las imagenes en el nuevo espacio
    for(int i=0;i<mathappy.rows;i++){
       Mat point,dst;
       point = pca_ha.project(mathappy.row(i));
       cv::normalize(point, dst, -1, 1, NORM_MINMAX,CV_32FC1); //Normalizacion entre -1 y 1
       pca_points.push_back(dst);
    }
    //---------------------------------------------------------------------------------------

    //Creo una Mat para poder aplicar PCA Sad
    cv::Mat matsad(sad.size(), sad.at(0).size(), CV_32FC1);
    for(int i=0; i<matsad.rows; ++i)
         for(int j=0; j<matsad.cols; ++j)
              matsad.at<float>(i, j) = sad.at(i).at(j);


    //PCA sobre datos Sad
    PCA pca_sa(matsad, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //Quiero saber la Varianza total retenida
    cout<<"numero de componentes sad: "<<pca_sa.eigenvalues.rows<<endl;

    //Proyecta todos las imagenes en el nuevo espacio
    for(int i=0;i<matsad.rows;i++){
       Mat point,dst;
       point = pca_sa.project(matsad.row(i));
       cv::normalize(point, dst, -1, 1, NORM_MINMAX,CV_32FC1); //Normalizacion entre -1 y 1
       pca_points.push_back(dst);
    }
    //---------------------------------------------------------------------------------------

    //Creo una Mat para poder aplicar PCA Surprise
    cv::Mat matsur(sur.size(), sur.at(0).size(), CV_32FC1);
    for(int i=0; i<matsur.rows; ++i)
         for(int j=0; j<matsur.cols; ++j)
              matsur.at<float>(i, j) = sur.at(i).at(j);


    //PCA sobre datos Surprise
    PCA pca_su(matsur, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //Quiero saber la Varianza total retenida
    cout<<"numero de componentes surprise: "<<pca_su.eigenvalues.rows<<endl;

    //Proyecta todos las imagenes en el nuevo espacio
    for(int i=0;i<matsur.rows;i++){
       Mat point,dst;
       point = pca_su.project(matsur.row(i));
       cv::normalize(point, dst, -1, 1, NORM_MINMAX,CV_32FC1); //Normalizacion entre -1 y 1
       pca_points.push_back(dst);
    }
    //---------------------------------------------------------------------------------------

    //Escribimos en archivo salida de pca_points
    std::string str=path;
    int position = str.find ('.');
    str.replace(position,4,"-pca-projections.csv");

    Mat an_di_fe_ha_sa_su_pca = asRowMatrix(pca_points,CV_32FC1);
    writeMatsToFile(an_di_fe_ha_sa_su_pca,str.c_str());

    cout<<endl<<"PCA aplicado con exito"<<endl;



}




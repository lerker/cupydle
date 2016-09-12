#include "functions.h"
#include <math.h>
#include <fstream>


using namespace std;
using namespace cv;

bool no_face=false;

IplImage* getCameraFrame(CvCapture* &camera, const char *filename = 0, int camid=0, int width=320, int height=240)
{
    IplImage *frame = 0;
    int w, h;

    // If the camera hasn't been initialized, then open it.
    if (!camera)
    {
        if (!filename)
        {
            //printf("Acessing the camera ...\n");
            camera = cvCaptureFromCAM(camid);
            // Try to set the camera resolution.
            cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH, width);
            cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_HEIGHT, height);
        } else {
            //printf("Acessing the video sequence ...\n");
            camera = cvCaptureFromAVI(filename);
        }

        if (!camera)
        {
            printf("Couldn't access the camera.\n");
            exit(1);
        }

        // Get the first frame, to make sure the camera is initialized.
        frame = cvQueryFrame( camera );
        //frame = cvRetrieveFrame(camera);

        if (frame)
        {
            w = frame->width;
            h = frame->height;
            //printf("Got the camera at %dx%d resolution.\n", w, h);
        }
        //sleep(10);
    }

    // Wait until the next camera frame is ready, then grab it.
    frame = cvQueryFrame( camera );
    //frame = cvRetrieveFrame(camera);
    if (!frame)
    {
        printf("Couldn't grab a camera frame.\n");
        return frame;
    }
    return frame;
}

void detectFaceInImage(IplImage *orig, IplImage* input, CvHaarClassifierCascade* cascade, FLANDMARK_Model *model, int *bbox, double *landmarks)
{
    // Smallest face size.
    CvSize minFeatureSize = cvSize(100, 100);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;

    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    // Detect all the faces in the greyscale image.
    rects = cvHaarDetectObjects(input, cascade, storage, search_scale_factor, 2, flags, minFeatureSize);
    nFaces = rects->total;

    double t = (double)cvGetTickCount();
    for (int iface = 0; iface < (rects ? nFaces : 0); ++iface)
    {
        CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);

        bbox[0] = r->x;
        bbox[1] = r->y;
        bbox[2] = r->x + r->width;
        bbox[3] = r->y + r->height;

        flandmark_detect(input, bbox, model, landmarks);
        // display landmarks
        //No me interesa dibujar los landmarks
        /*
        cvRectangle(orig, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255,0,0) );
        cvRectangle(orig, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
        cvCircle(orig, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0, 0,255), CV_FILLED);
        for (int i = 2; i < 2*model->data.options.M; i += 2)
        {
            cvCircle(orig, cvPoint(int(landmarks[i]), int(landmarks[i+1])), 3, CV_RGB(255,0,0), CV_FILLED);

        }*/
    }
    t = (double)cvGetTickCount() - t;
    int ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );

    if (nFaces > 0)
    {
        no_face=false;
        //printf("Faces detected: %d; Detection of facial landmark on all faces took %d ms\n", nFaces, ms);
    } else {
        no_face=true;
        printf("NO Face\n");
    }

    cvReleaseMemStorage(&storage);
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

void writeMatsToFile(cv::Mat& m, cv::Mat& m2, const char* filename)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<" ";
        }
        for(int j=0; j<m2.cols; j++)
        {
            fout<<m2.at<float>(i,j)<<" ";
        }

        fout<<endl;
    }

    fout.close();
}

void extrae_caract(char* path,cv::Mat & datos_boca, cv::Mat & datos_ojos)
{
    char flandmark_window[] = "region_boca";
    char flandmark_window2[] = "region_ojos";
    double t;
    int ms;

    const char *infname = 0;

    int vidfps, frameW, frameH, fourcc, nframes = 0;
    //int fourcc = CV_FOURCC('D', 'I', 'V', 'X');

    CvCapture* camera = 0;	// The camera device.
    IplImage *frame = 0;

    infname = path;
    //printf("infname = %s\n", infname);

    frame = getCameraFrame(camera, infname);
    frameH = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_HEIGHT);
    frameW = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH);
    fourcc = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FOURCC);
    nframes = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_COUNT);
    vidfps = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FPS);


    cvNamedWindow(flandmark_window, 0);
    // Haar Cascade file, used for Face Detection.
    char faceCascadeFilename [] = "/home/ramiro/Repositorios/pfc/code/VideoFeaturesExtraction/haarcascade_frontalface_alt.xml";
    // Load the HaarCascade classifier for face detection.
    CvHaarClassifierCascade* faceCascade;
    faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);
    if( !faceCascade )
    {
        printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
        exit(1);
    }

    // ------------- begin flandmark load model
    t = (double)cvGetTickCount();
    FLANDMARK_Model * model = flandmark_init("/home/ramiro/Repositorios/pfc/code/VideoFeaturesExtraction/flandmark_model.dat");
    if (model == 0)
    {
        printf("Structure model was not created. Corrupted file flandmark_model.dat?\n");
        exit(1);
    }
    t = (double)cvGetTickCount() - t;
    ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
    //printf("Structure model loaded in %d ms.\n", ms);
    // ------------- end flandmark load model

    int *bbox = (int*)malloc(4*sizeof(int));
    double *landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);

    char fps[50];
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);

    //cout<<"Numero de frames "<<nframes<<endl;

   //defino p para contar
    int p=0;

    //Defino vector para guardar frames relevantes y luego poder aplicar PCA
    vector<Mat> frames_boca;
    vector<Mat> frames_ojos;

    //Abrimos archivo que contiene frame inicial y final de video donde el sujeto habla
    std::string name(path);
    string str2 = ".avi";
    name.replace(name.find(str2),str2.length(),"-voiced_frames.txt");
    ifstream fin(name.c_str());
    string s1,s2;
    getline(fin,s1,',');
    getline(fin,s2,'\n');

    int frame_ini = atoi(s1.c_str());
    int frame_fin = atoi(s2.c_str());
    //cout<<frame_ini<<" "<<frame_fin<<endl;
    bool repeat_middle = false;
    int frames_sobra;
    int frames_falta;
    int middle_frame;

    // Con un script afuera determinamos que el promedio de frames relevantes es 47
    int frames_prom = 47;
    // Tomamos entonces 47 frames para todos los videos
    if ((frame_fin-frame_ini)>frames_prom){

        frames_sobra = (frame_fin-frame_ini)-frames_prom; // frames extra
        // si frames_sobra es 5, a frame_ini le sumamos 2 y a frame_fin le restamos 3
        //cout<<"Frames de sobra: "<<frames_sobra<<endl;
        int suma_i=frames_sobra/2;
        if(frames_sobra%2==0){
            frame_ini += suma_i;
            frame_fin -= suma_i;
        }
        else{
            frame_ini += suma_i+1;
            frame_fin -= suma_i;
        }
        //cout<<suma_i<<endl;

    }
    else if((frame_fin-frame_ini)<frames_prom){
        frames_falta = frames_prom-(frame_fin-frame_ini);
        middle_frame = floor(((float)(frame_fin-frame_ini))/2)+frame_ini;
        repeat_middle = true;
    }
    // Este for es para descartar los frames iniciales donde no hay voz
    for(int i=0;i<frame_ini;i++){
      frame = getCameraFrame(camera, infname);
    }

    //Recorremos frames de video donde hay voz
    for (int i=frame_ini;i<frame_fin;i++){
        //cout<<"i= "<<i<<endl;
        // si es necesario, repetimos el frame del medio tantas veces como frames falten para 47
        if(repeat_middle){
            if (frames_falta>0){
                if(i==middle_frame){
                    i--;
                    frames_falta--;
                }
            }
        }
        t = (double)cvGetTickCount();
        frame = getCameraFrame(camera, infname);

        cvConvertImage(frame, frame_bw);
        detectFaceInImage(frame, frame_bw, faceCascade, model, bbox, landmarks);

        //mostrar
        t = (double)cvGetTickCount() - t;
        sprintf(fps, "%.2f fps", 1000.0/( t/((double)cvGetTickFrequency() * 1000.0) ) );
        cvPutText(frame, fps, cvPoint(10, 40), &font, cvScalar(255, 0, 0, 0));
        cvShowImage(flandmark_window, frame);
        //cvShowImage(flandmark_window,salida);
        // imshow(flandmark_window,salida_boca);
        //imshow(flandmark_window2,salida_ojos);
        cvWaitKey(100);

        cout<<p<<endl;
        //Defino imagen de salida para la region de la boca

        cv::Mat salida_boca;
        cv::Mat salida_ojos;
        //recorto cada frame segun landmarks
        region_boca(frame,landmarks,salida_boca);
        region_ojos(frame,landmarks,salida_ojos);

        //Paso a escala de grises
        cvtColor(salida_boca, salida_boca, CV_RGB2GRAY);
        cvtColor(salida_ojos, salida_ojos, CV_RGB2GRAY);
        //Guardo
        frames_boca.push_back(salida_boca);
        frames_ojos.push_back(salida_ojos);

        p++;//avanza a sig. frame

     }

     //cout<<"frames relevantes: "<<p<<endl;

    // Free the camera.
    free(landmarks);
    free(bbox);
    cvReleaseCapture(&camera);
    cvReleaseHaarClassifierCascade(&faceCascade);
    cvDestroyWindow(flandmark_window);
    flandmark_free(model);
    //cout<<"Llego"<<endl;

    datos_boca = asRowMatrix(frames_boca, CV_32FC1);
    datos_ojos = asRowMatrix(frames_ojos, CV_32FC1);

    //cout<<"size datos_boca :"<<datos_boca.rows<<"x"<<datos_boca.cols<<endl;
    //cout<<"size datos_ojos :"<<datos_ojos.rows<<"x"<<datos_ojos.cols<<endl;

}

void region_boca(IplImage *input, double *landmarks, cv::Mat & output){
    //Extremo izq y der de la boca y nariz
    float pto_izq_x=landmarks[6];
    float pto_izq_y=landmarks[7];
    float pto_der_x=landmarks[8];
    float pto_der_y=landmarks[9];
    float pto_nar_x=landmarks[14];
    float pto_nar_y=landmarks[15];

    //Convertimos input tipo Iplimage a cv::Mat
    cv::Mat image(input,false);
    //Recortamos
    int delta_x=60;
    int w=(pto_der_x-pto_izq_x) + delta_x;
    int h=(pto_der_y-pto_nar_y)*2;
    cv::Rect zona_boca(pto_izq_x-(delta_x/2),pto_izq_y-h/2,w,h);
    output = image(zona_boca).clone();
    //Escalamos imagen de salida
    cv::Size tamanio(70,35);
    resize(output, output, tamanio, 0, 0);

}

void region_ojos(IplImage *input, double *landmarks, cv::Mat & output){
    //Extremo izq de ojo izq y extremo derecho ojo derecho
    float pto_izq_x=landmarks[10];
    float pto_izq_y=landmarks[11];
    float pto_der_x=landmarks[12];
    float pto_der_y=landmarks[13];
    float pto_cen_x=landmarks[14];
    float pto_cen_y=landmarks[15];

    //Convertimos input tipo Iplimage a cv::Mat
    cv::Mat image(input,false);
    //Recortamos
    int delta_x=30,delta_y=20;
    int w=(pto_der_x-pto_izq_x) + delta_x;
    int h=(pto_cen_y-pto_izq_y) + delta_y;
    cv::Rect zona_ojos(pto_izq_x-(delta_x/2),pto_izq_y-h/1.5,w,h);
    output = image(zona_ojos).clone();
    //Escalamos imagen de salida
    cv::Size tamanio(70,35);
    resize(output, output, tamanio, 0, 0);

}



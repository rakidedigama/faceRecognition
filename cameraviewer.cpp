#include "cameraviewer.h"
#include "VideoFaceDetector.h"
#include <QThread>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include "iostream";
#include "QDateTime"

using namespace cv;
using namespace std;

const cv::String    WINDOW_NAME("Camera video");
const cv::String    CASCADE_FILE = "C:/BUILDS/openCV/install/etc/haarcascades/haarcascade_frontalface_alt.xml";

//VideoCapture camera("rtsp://192.168.1.24/profile2/media.smp");
//VideoCapture camera("rtsp://192.168.1.64/profile2/media.smp");
VideoCapture camera;

//std::cout<<"openign camera";

//VideoFaceDetector detector(CASCADE_FILE, camera);


CameraViewer::CameraViewer(QObject *parent):
    QThread(parent)
{
    sendSignal = false;
    foundCamera = false;
    saveVideo = false;
}

void CameraViewer::run(){

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO | cv::WINDOW_AUTOSIZE);
    //cv::resizeWindow(WINDOW_NAME,640,360);
    // Web Camera
    const string videoStreamAddress1 = "rtsp://192.168.1.7/profile2/media.smp";
    const string videoStreamAddress2 = "rtsp://192.168.1.24/profile2/media.smp";
    const string videoStreamAddress3 = "rtsp://192.168.1.64/profile2/media.smp";


    const QDateTime videoName = QDateTime::currentDateTime();
    const QString timestamp = videoName.toString(QLatin1String("yyyyMMdd-hhmmss"));
    const QString fileName = QString::fromLatin1("C:/BUILDS/FaceDetectionProject/Work2/Videos/clip-%1.avi").arg(timestamp);

    while(!foundCamera){
        //if (!camera.open(videoStreamAddress3)){
       if (!camera.open(0)) {
       // if (!camera.grab()){
            cout<<"Error getting camera..\n";
        }
        else{
            foundCamera = true;     }
    }

    int ex = static_cast<int>(camera.get(CV_CAP_PROP_FOURCC));
    Size frameSize(static_cast<int>(1280), static_cast<int>(720));
     VideoWriter oVideoWriter (fileName.toStdString(), CV_FOURCC('M','J','P','G'), 15, frameSize,true);

    while(foundCamera){      
        gotRectFace = false;
        camera>>m_frame;

        if(!m_frame.empty()){        //
            if(saveVideo){
                //cout<<"Savevideo true" <<endl;
                  //cout<<fileName.toStdString()<<endl;

               if ( !oVideoWriter.isOpened() )
               {
                   cout << "ERROR: Failed to write the video" << endl;
                }
               Mat saveFrame;
               cv::resize(m_frame,saveFrame,Size(1280,720));
               oVideoWriter<<saveFrame;
            }

            emit sendFrame(m_frame);//full frame

            if(gotRectFace){
              cv::rectangle(m_frame,faceRect, cv::Scalar(255, 0, 0));
            }
        }
        cv::imshow(WINDOW_NAME, m_frame);
        if (cv::waitKey(25) == 27)
        break;
    }
}

void CameraViewer::catchFaceRectangle(cv::Rect mface){
    gotRectFace = true;
    faceRect = mface;
}

void CameraViewer::saveVideos(bool value){
    saveVideo = value;
}



#ifndef CAMERAVIEWER_H
#define CAMERAVIEWER_H


#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "VideoFaceDetector.h"

#include <QThread>

using namespace cv;

class CameraViewer : public QThread
{
    Q_OBJECT
public:
    explicit CameraViewer(QObject *parent = 0);

protected:
    virtual void run();
    bool sendSignal;
    bool foundCamera;
    bool gotRectFace;
    bool saveVideo;
   cv::Rect faceRect;
    //VideoCapture m_camera;
    //VideoFaceDetector m_detector;
    cv::Mat m_frame;

signals:
    void sendFrame(cv::Mat);

public slots:
    void catchFaceRectangle(cv::Rect mface);
    void saveVideos(bool value);

//public slots:
//    void getDetectedFace();



};

#endif // CAMERAVIEWER_H

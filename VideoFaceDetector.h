#pragma once

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\objdetect.hpp>
#include "qstring.h"
#include <QThread>

class VideoFaceDetector :  public QThread
{
    Q_OBJECT
public:
    VideoFaceDetector(const std::string cascadeFilePath);
    ~VideoFaceDetector();

    cv::Point               getFrameAndDetect();
    cv::Point               operator>>(cv::Mat &frame);
    void                    setVideoCapture(cv::VideoCapture &videoCapture);
    cv::VideoCapture*       videoCapture() const;
    void                    setFaceCascade(const std::string cascadeFilePath);
    cv::CascadeClassifier*  faceCascade() const;
    void                    setResizedWidth(const int width);
    int                     resizedWidth() const;
	bool					isFaceFound() const;
    cv::Rect                face(const cv::Mat &frame);
    cv::Point               facePosition() const;
    void                    setTemplateMatchingMaxDuration(const double s);
    double                  templateMatchingMaxDuration() const;
    //void                    saveFace(QString imageName);
    cv::Mat                 getFaceTemplate(const cv::Mat &frame, cv::Rect face);
    cv::Mat                 getROIImage();      // facesize color image
    cv::Mat                 getFaceImage(); // grey level pgm of roi image
    cv::Mat                 getTrackedFace(); // trackedface ->biggestface
protected:
    virtual void run();
signals:
    void sendFace(cv::Mat);
    void sendRectangle(cv::Rect);

  public slots:
    void catchFrame(cv::Mat);
private:
    static const double     TICK_FREQUENCY;

    cv::VideoCapture*       m_videoCapture;
    cv::CascadeClassifier*  m_faceCascade;
    std::vector<cv::Rect>   m_allFaces;
    cv::Rect                m_trackedFace;
    cv::Rect                 m_faceRoi;
    cv::Mat                 m_faceTemplate;
    cv::Mat                 m_matchingResult;
    bool                    m_templateMatchingRunning;
    int64                   m_templateMatchingStartTime;
    int64                   m_templateMatchingCurrentTime ;
    bool                    m_foundFace ;
    double                  m_scale;
    int                     m_resizedWidth;
    cv::Point               m_facePosition;
    double                  m_templateMatchingMaxDuration;
    cv::Mat                 m_face;

    cv::Mat                 frame;

    cv::Rect    doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const;
    cv::Rect    biggestFace(std::vector<cv::Rect> &faces) const;
    cv::Point   centerOfRect(const cv::Rect &rect) const;

    void        detectFaceFullImage(const cv::Mat &frame);
    void        detectFaceAroundRoi(const cv::Mat &frame);
    void        detectFacesTemplateMatching(const cv::Mat &frame);
};


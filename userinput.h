#ifndef USERINPUT_H
#define USERINPUT_H

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2\objdetect.hpp>
#include "opencv2/imgproc.hpp"
#include <QThread>
#include <QDir>
#include <QVector>
#include "faceclassifier.h"
#include "congealer.h"

using namespace cv;
using namespace std;
using namespace face;
class UserInput : public QThread
{
    Q_OBJECT
public:
    explicit UserInput(QObject *parent=0);
    void saveImages();
    void congealImages();
    int  predictImages(); // return int label
    void trainImages(int num);
    void loadModel();
    void UserInput::getUserInput();
    cv::Mat getCurrentFace();
    void writeNameToFile(QString name);
    void UserInput::readNames();
    void UserInput::printNames();
    void UserInput::createClassifier();
    void UserInput::runCrossTest();
   void UserInput::loadImagesFromFolder(string fpath,int num);
   void UserInput::assignLabels(string fpath);
   void UserInput::readLabels(string fpath,int num);
   void UserInput::outConfidence(double err, int o_label, int p_label); // write to file
    void UserInput::output();
    void UserInput::alignFace(Mat faceImage, int i);
    bool UserInput::eyesFound(Mat faceImage);
    void UserInput::setEyeCascade(const std::string eyeCascadeFilePath);

protected:
    void virtual run();
    QString m_action;
    bool m_gotFrame;

    cv::Mat m_frame;
    cv::Mat m_face;
    std::string m_location;
    std::string testLocation;
    QString nameList;

    QStringList m_names;
    QFile fNames;
    QFile outFile;
    string folderPath;
    string crossTestFolder;
    string eyeCascadePath;

    bool save;
    bool train;
    bool load;
    bool predict;
    bool crossTest;
    bool saveVideo;
//    int classRate;
    QVector<int> personHitVector;
    QVector<int> ctpArray;
    bool gotFace;
    int gotFaceCount; // to sum total number of face frames

    int person; // number of people or saved folders
    int faceCount;
    int numberOfFolders;


    vector<Mat> m_images;
    vector<int> m_labels;

    FaceClassifier m_classifier;
    //cv::CascadeClassifier* eyeCascade;


public slots:

    void catchFace(cv::Mat pFace);
signals:
    void congealImages(QString inputFile, QString folder, bool congSignal);
    void requestFace();
    void savingFacesMode(bool value);




};

#endif // USERINPUT_H

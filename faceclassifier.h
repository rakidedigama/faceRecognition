#ifndef FACECLASSIFIER_H
#define FACECLASSIFIER_H

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace face;
class FaceClassifier
{

public:

    FaceClassifier::FaceClassifier(Ptr<BasicFaceRecognizer> model);
    void FaceClassifier::setClassifier(Ptr<BasicFaceRecognizer> model);
    void FaceClassifier::updateThreshold(double newThreshold);
    void FaceClassifier::trainClassifier();
    void FaceClassifier::trainClassifier(vector<Mat> images,vector<int> labels);
    void FaceClassifier::saveModel();
    void FaceClassifier::loadModel();
    int FaceClassifier::predictLabel(Mat predImg);
    Mat FaceClassifier::getMeanFace();
    Mat FaceClassifier::getEigenVectors();
    Mat FaceClassifier::getEigenValues();
    double FaceClassifier::getProjectedDifference(Mat unknownFace);
    double FaceClassifier::getConfidence(Mat predImg,int pLabel);
   vector<int> FaceClassifier::getTrainingLabels();
    vector<Mat> FaceClassifier::getTrainingImages();

    static Mat norm_0_255(InputArray _src);
private:


    vector<Mat> m_images;
    vector<int> m_labels;
    Ptr<BasicFaceRecognizer> m_model ;



};

#endif // FACECLASSIFIER_H

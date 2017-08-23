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

    FaceClassifier::FaceClassifier(Ptr<FaceRecognizer> model);
    void FaceClassifier::setClassifier(Ptr<FaceRecognizer> model);
    void FaceClassifier::trainClassifier();
    void FaceClassifier::trainClassifier(vector<Mat> images,vector<int> labels);
    int FaceClassifier::predictLabel(Mat predImg);
    double FaceClassifier::getConfidence(Mat predImg,int pLabel);
   vector<int> FaceClassifier::getTrainingLabels();
    vector<Mat> FaceClassifier::getTrainingImages();
private:
    vector<Mat> m_images;
    vector<int> m_labels;
    Ptr<FaceRecognizer> m_model ;



};

#endif // FACECLASSIFIER_H

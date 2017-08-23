#include "faceclassifier.h"
#include "iostream"
#include "fstream"
#include "istream"
#include "sstream"

using namespace std;
using namespace cv;
using namespace face;

FaceClassifier::FaceClassifier(Ptr<FaceRecognizer> model)
{
    cout<<"Created Classifier"<<endl;
//    Ptr<FaceRecognizer> model = EigenFaceRecognizer::create();
//    vector<Mat> m_images;
//    vector<int> m_labels;
    setClassifier(model);

}
void FaceClassifier::setClassifier(Ptr<FaceRecognizer> model)
{
    m_model = model;
}

void FaceClassifier::trainClassifier()
{
    m_model->train(m_images,m_labels);
}

void FaceClassifier::trainClassifier(vector<Mat> imagesIn,vector<int> labelsIn)
{
    m_images = imagesIn;
    m_labels = labelsIn;
    cout<<"In TrainClassifier ";
    cout<<" Number of Images: " << m_images.size();
    cout<<" Number of Labels " << m_labels.size()<<endl;
    m_model->train(m_images,m_labels);
    cout<<"Training Done"<<endl;
}

int FaceClassifier::predictLabel(Mat predImg )
{
   int predictedLabel = m_model->predict(predImg);
    return predictedLabel;
}
double FaceClassifier::getConfidence(Mat predImg,int pLabel)
{
    double predicted_confidence;
    m_model->predict(predImg, pLabel, predicted_confidence);
    return predicted_confidence;
}

vector<Mat> FaceClassifier::getTrainingImages()
{

    return m_images;
}

vector<int> FaceClassifier::getTrainingLabels()
{
    return m_labels;
}




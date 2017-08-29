#include "faceclassifier.h"
#include "iostream"
#include "fstream"
#include "istream"
#include "sstream"

using namespace std;
using namespace cv;
using namespace face;

Mat FaceClassifier::norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}


FaceClassifier::FaceClassifier(Ptr<BasicFaceRecognizer> model)
{
    cout<<"Created Classifier"<<endl;
//    Ptr<FaceRecognizer> model = EigenFaceRecognizer::create();
//    vector<Mat> m_images;
//    vector<int> m_labels;
    setClassifier(model);

}
void FaceClassifier::setClassifier(Ptr<BasicFaceRecognizer> model)
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
void FaceClassifier::saveModel(){
    m_model->save("eigenFaces_t.yml");
}
void FaceClassifier::loadModel(){
//     m_model->load("eigenFaces_t.yml");
    //const String name = "eigenFaces_t.yml";
    // m_model->loadFromString(name);
     //m_model->load(&name);

    Ptr<EigenFaceRecognizer> yml = Algorithm::load<EigenFaceRecognizer>("eigenFaces_t.yml");
    setClassifier(yml);

}

void FaceClassifier::updateThreshold(double newThreshold){

    //double current_threshold = model->getDouble("threshold");
    // And this line sets the threshold to 0.0:
    m_model->setThreshold(newThreshold);
           //("threshold", newThreshold);
}

int FaceClassifier::predictLabel(Mat predImg )
{
   int predictedLabel = m_model->predict(predImg);
    return predictedLabel;
}
Mat FaceClassifier::getEigenVectors(){
    Mat eigenVectors = m_model->getEigenVectors();
    return eigenVectors;
}

// Eigenvalues (ratios) for mean image
Mat FaceClassifier::getEigenValues(){
    Mat eigenValues = m_model->getEigenValues();
    for(int i=0;i<eigenValues.rows;i++){
        cout<<eigenValues.row(i);
    }
    return eigenValues;
}
Mat FaceClassifier::getMeanFace(){
    Mat mean = m_model->getMean();
    return mean;
}

double FaceClassifier::getProjectedDifference(Mat unknownFace){
    // Project the input face onto the eigenspace.
    Mat projection = LDA::subspaceProject(m_model->getEigenVectors(), m_model->getMean(),unknownFace.reshape(1,1));
    //Generate the reconstructed face
    Mat reconstructionRow = LDA::subspaceReconstruct(m_model->getEigenVectors(), m_model->getMean(),projection);

    // Reshape the row mat to an image mat
   /* Mat reconstructionMat = reconstructionRow.reshape(1,unknownFace.rows);
    // Convert the floating-point pixels to regular 8-bit uchar.
    Mat reconstructed_face = Mat(reconstructionMat.size(), CV_8U);

    reconstructionMat.convertTo(reconstructed_face, CV_8U, 1, 0);*/

    Mat reconstructionMat = FaceClassifier::norm_0_255(reconstructionRow.reshape(1, unknownFace.rows));
    double err = norm(unknownFace,reconstructionMat, CV_L2);
    //convert to scale
    //double similarity = error / (double)(input_face.rows * input_face.cols);

    return err;
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






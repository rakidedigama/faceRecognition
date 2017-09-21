/* Things to do
 * - Write label names to text file and read
 * - Face detection accuracy - only save once face is found
 * - new face to others based on confidence value
 */


#include <QCoreApplication>
#include <QConsoleDebugStream.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
//#include <stdio.h>
#include <QDebug>
//#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "cameraviewer.h"
#include "faceclassifier.h"
#include "VideoFaceDetector.h"
#include <QTime>
#include <QDir>
#include <QStringList>
#include "userinput.h"

#include <QFuture>
#include <QtConcurrent/QtConcurrent>

using namespace cv;
using namespace std;

bool fullProgram =true ;




string name;
QStringList names;
QString workFolder = "C:/BUILDS/FaceDetectionProject/Work/";
string savedModelFile = "eigenFaces_t.yml";


const cv::String    WINDOW_NAME("Camera video");
const cv::String    CASCADE_FILE = "C:/BUILDS/openCV/install/etc/haarcascades/haarcascade_frontalface_alt.xml";



static void readTrainingData_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }

    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}



int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    //Q_ConsoleDebugStream stream(std::cout, "log_DalsaImageAnalyzer");
    CameraViewer c;  // grab frames and send to faceDetector
    VideoFaceDetector d(CASCADE_FILE); // detect faces and send to User

    UserInput u; // user program

    qRegisterMetaType< cv::Mat >("cv::Mat");
    qRegisterMetaType< cv::Rect >("cv::Rect");
    QObject::connect(&c,SIGNAL(sendFrame(cv::Mat)),&d,SLOT(catchFrame(cv::Mat)));
    QObject::connect(&d,SIGNAL(sendRectangle(cv::Rect)),&c,SLOT(catchFaceRectangle(cv::Rect)));

    QObject::connect(&d,SIGNAL(sendFace(cv::Mat)),&u,SLOT(catchFace(cv::Mat)));
   // QObject::connect(&c,SIGNAL(sendDetectedFace(cv::Mat)),&u,SLOT(catchFace(cv::Mat)));
    QObject::connect(&u,SIGNAL(savingFacesMode(bool)),&c,SLOT(saveVideos(bool)));

    cout<<"Starting camera"<<endl;
    c.start();
    cout<<"Starting detector" <<endl;
    d.start();
    cout<<"starting user"<<endl;
    u.start();





/*
    while (true)
    {

        int presentTime = myTimer.elapsed();
        int timeDelay = presentTime - previousTime;
        if(userInput== ""){
            cout << "Enter input: s-save images, t-train, p-predict : ";
            cin >> userInput;
            if(userInput=="s"){
                saveImages = true;
                string folderPath = "C:/BUILDS/FaceDetectionProject/Work/commonArea/";
                QDir dir(QString::fromStdString(folderPath));
                cout<<"Number of people: " << dir.count()-2;
                cout<<"Enter name of person: ";
                string name;
                cin >> name;
                person = dir.count()-2 +1; // dir.Count() returns number of files + 2. Subtract names.txt file
                cout<<"Number of people updated : "<< person<<" set SaveImges : True; "<<" Writing name to File and adding to list: " << name;
                writeNameToFile(QString::fromStdString(name));  // write name to local text file
                names.append(QString::fromStdString(name));

                u.saveImages();

            }
            if(userInput=="x"){
                saveImages = false;
                loadFromXml = true;
                cout<<"Loading Training data from XML"; }
            if(userInput=="t"){
                trainImages = true;
                cout<<"Start Training Images" <<endl; }
            if(userInput=="l"){
                loadModel = true;
                cout<<"Loading Model" << endl; }
            if(userInput=="p"){
                predict = true;
                cout<<"Start Recognition" <<endl; }
        }



    if(saveImages){
        if (faceCount<=maxImages && timeDelay>200) {
            string subFolderPath = format("C:/BUILDS/FaceDetectionProject/Work/commonArea/s%d",person);///%d.pgm,faceCount
            QDir dirSub(QString::fromStdString(subFolderPath));
            if (!dirSub.exists()) {  // Create Subfolder
                dirSub.mkpath(QString::fromStdString(subFolderPath));      }
            string fileName = format("%d.pgm",faceCount);
            QString imageName =  QString::fromStdString(subFolderPath) + "/" +  QString::fromStdString(fileName) ;


             cv::Mat faceImageGrayResized = u.getCurrentFace();
            cv::imwrite(imageName.toStdString(),faceImageGrayResized);
            cout<<"saving Face Image"<< imageName.toStdString()<<endl;
            //printf("Saving Face Image\n");

            //detector.saveFace(imageName);
            previousTime = myTimer.elapsed();
            faceCount++;       }
        if(faceCount>maxImages)
        {
            cout<< maxImages << " Images already saved" <<endl;
            numberOfPersons++;
            saveImages = false;
            faceCount = 1;              // resest faceCount to 0 for new person
            userInput = "";
        }
    }
    }*/


    return a.exec();
}



   // u.viewFrames();





    /*
    //Q_ConsoleDebugStream stream(std::cout, "log_DalsaImageAnalyzer");
    VideoCapture camera;
    cout<<"openign camera";

//    std::string videoStreamAddress = "rtsp://192.168.1.7/profile2/media.smp";
//    if (!camera.open(videoStreamAddress)) {
//        //fprintf(stderr, "Error getting camera...\n");
//        cout<<"Error getting camera..\n";
//        exit(1);
//    }

    // Web Camera
    if (!camera.open(0)) {
        fprintf(stderr, "Error getting camera...\n"); exit(1);
    }

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO | cv::WINDOW_AUTOSIZE);
    VideoFaceDetector detector(CASCADE_FILE, camera);
    cv::Mat frame;

    Ptr<BasicFaceRecognizer> model1 = EigenFaceRecognizer::create(eigenFaces);
    FaceClassifier classifier(model1);

    vector<Mat> images;             // Empty vector
    vector<qint32> labels;
    readNames();
    cout <<"Priniting names already registered : " <<names.size()<<endl;
     for(int i=0;i<names.size();i++){
         cout<<names[i].toStdString()<<endl;
     }

    myTimer.start();
    int previousTime = 0;
    int person = 0;
    int faceCount = 1;
    string userInput = "";
    double fps = 0, time_per_frame;


    while (true)
    {
        auto start = cv::getCPUTickCount();
        detector >> frame;
        auto end = cv::getCPUTickCount();
        time_per_frame = (end - start) / cv::getTickFrequency();
        fps = (15 * fps + (1 / time_per_frame)) / 16;

        if(userInput== ""){
            cout << "Enter input: s-save images, t-train, p-predict : ";
            cin >> userInput;
            if(userInput=="s"){
                saveImages = true;
                string folderPath = "C:/BUILDS/FaceDetectionProject/Work/commonArea/";
                QDir dir(QString::fromStdString(folderPath));
                cout<<"Number of people: " << dir.count()-2;
                cout<<"Enter name of person: ";
                string name;
                cin >> name;
                person = dir.count()-2 +1; // dir.Count() returns number of files + 2. Subtract names.txt file
                cout<<"Number of people updated : "<< person<<" set SaveImges : True; "<<" Writing name to File and adding to list: " << name;
                writeNameToFile(QString::fromStdString(name));  // write name to local text file
                names.append(QString::fromStdString(name));    }
            if(userInput=="x"){
                saveImages = false;
                loadFromXml = true;
                cout<<"Loading Training data from XML"; }
            if(userInput=="t"){
                trainImages = true;
                cout<<"Start Training Images" <<endl; }
            if(userInput=="l"){
                loadModel = true;
                cout<<"Loading Model" << endl; }
            if(userInput=="p"){
                predict = true;
                cout<<"Start Recognition" <<endl; }
        }

        if (detector.isFaceFound())
        {
            printf("Time per frame: %3.3f\tFPS: %3.3f\n", time_per_frame, fps);
            cv::rectangle(frame, detector.face(frame), cv::Scalar(255, 0, 0));
            int presentTime = myTimer.elapsed();
            int timeDelay = presentTime - previousTime;

            if(saveImages){
                if (faceCount<=maxImages && timeDelay>200) {
                    string subFolderPath = format("C:/BUILDS/FaceDetectionProject/Work/commonArea/s%d",person);///%d.pgm,faceCount
                    QDir dirSub(QString::fromStdString(subFolderPath));
                    if (!dirSub.exists()) {  // Create Subfolder
                        dirSub.mkpath(QString::fromStdString(subFolderPath));      }
                    string fileName = format("%d.pgm",faceCount);
                    QString imageName =  QString::fromStdString(subFolderPath) + "/" +  QString::fromStdString(fileName) ;
                    cout<<imageName.toStdString()<<endl;
                    detector.saveFace(imageName);
                    previousTime = myTimer.elapsed();
                    faceCount++;       }
                if(faceCount>maxImages)
                {
                    cout<< maxImages << " Images already saved" <<endl;
                    numberOfPersons++;
                    saveImages = false;
                    faceCount = 1;              // resest faceCount to 0 for new person
                    userInput = "";
                }
            }
            if(loadFromXml){
                QString fn_csv = "C:\\BUILDS\\FaceDetectionProject\\faceImagesCSV.csv";
              try {
                  readTrainingData_csv(fn_csv.toStdString(), images, labels);
                  cout<< "Number of images: " << images.size();
                  cout<< "Number of labels: " << labels.size();
              } catch (cv::Exception& e) {
                  cerr << "Error opening csv file \"" << fn_csv.toStdString() << "\". Reason: " << e.msg << endl;
                  // nothing more we can do
                  cout<<"CSV reading error";
                  qDebug()<<"CSV reading error";
                  exit(1);
              }
                userInput ="";
            }

            if(predict){

                cv::Mat unknownFace = detector.getFaceImage(); // return grey pgm size image
                 //cv::Mat unknownFace = cv::imread("C:/BUILDS/FaceDetectionProject/commonArea/s2/6.pgm",CV_LOAD_IMAGE_GRAYSCALE);

                int predicted_label = -1;
                double predicted_confidence = 0.0;
                // Get the prediction and associated confidence from the model

                double err = classifier.getProjectedDifference(unknownFace);
                if(!unknownFace.empty()){
                    predicted_label = classifier.predictLabel(unknownFace);
                    predicted_confidence = classifier.getConfidence(unknownFace,predicted_label);
                    QString predName = names.at(predicted_label-1);
                    cout<< "predicted, " << predicted_label <<"Name : " << predName.toStdString()<<endl;
                    //<<" Confidence: " << predicted_confidence;
                    //cout<< "predicted, " << predicted_label <<endl;
                    //cout<<"Projected difference: "<< err;

                    if (outFile.open(QIODevice::WriteOnly | QIODevice::Append)) {
                      QTextStream s(&outFile);
                        s <<QDateTime::currentDateTime().toString()<<'\t'<< err << '\n';
                    } else {
                      std::cerr << "error opening output file\n";           }
                    outFile.close();          }
                else{
                    cout<<"Captured image empty" << endl;   }
            }

            //cv::circle(frame, detector.facePosition(), 30, cv::Scalar(0, 255, 0));
        }
        if(loadModel){
            classifier.loadModel();
            loadModel = false;
            userInput = "";
            cv::Mat vectors = classifier.getEigenVectors();
            cv::Mat values = classifier.getEigenValues();
            cout<<"Size of EigenVectors " << vectors.size;
            cout<<"Size of EigenValues " << values.size; }

        if(trainImages){
                //load images & labels externally from Save Folder
            if(!loadFromXml){
                cout<<"loading from folder";
                images = loadImagesFromFolder(images);
                labels = loadLabels(labels);         }
             loadFromXml = false;

             testSample = images[images.size() - 1];
             testLabel = labels[labels.size() - 1];
             images.pop_back();
             labels.pop_back();

             classifier.trainClassifier(images,labels);
             trainImages = false;
             userInput = "";
             cout<<"Total Number of images: " << images.size()<< " Number of labels: " << labels.size()<<" Size of image sample " << images[2].size();
             cout<<"Number of classes/folders/people: "<< numberOfFolders<< "Number of names in list: " << names.size()<<endl;
             for(int i = 0;i<names.size();i++){
             cout<<"Names are : " << names.at(i).toStdString()<<endl;             }
             //classifier.updateThreshold(1500);
             // Display properties of EigenVectors(faces) & values. Mean Face
             //imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
             cv::Mat vectors = classifier.getEigenVectors();
             cv::Mat values = classifier.getEigenValues();
             cout<<"Size of EigenVectors " << vectors.size;
             cout<<"Size of EigenValues " << values.size;

             classifier.saveModel();

        }
        else{
            cout<<"Face not found" << endl;      }

        cv::imshow(WINDOW_NAME, frame);
        if (cv::waitKey(25) == 27) break;

    }   //end detection




}*/

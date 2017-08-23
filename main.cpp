/* Things to do
 * - Write label names to text file and read
 * - Face detection accuracy - only save once face is found
 * - new face to others based on confidence value
 */


#include <QCoreApplication>
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

#include "faceclassifier.h"
#include "VideoFaceDetector.h"
#include <QTime>
#include <QDir>
#include <QStringList>

using namespace cv;
using namespace std;

bool fullProgram = true;
bool classificationTest = false;

// Stages in fullProgram - save or load training images from XML/train/predict
bool trainImages = false;
bool saveImages = false;
bool loadFromXml = false;
bool predict = false;
QTime myTimer;
cv::Mat faceImageGray;
cv::Mat faceImageGrayResized;

Mat testSample;
int testLabel ;

int numberOfPersons = 0;
int numberOfFolders = 0;
int maxImages = 50;
string name;
QStringList names;

QFile fIn("names.txt");

const cv::String    WINDOW_NAME("Camera video");
const cv::String CASCADE_FILE = "C:/BUILDS/openCV/install/etc/haarcascades/haarcascade_frontalface_alt.xml";

static void writeNameToFile(QString name){

    // write data
    QFile fOut("names.txt");
     //QIODevice fOut("names.txt");
     //f.open(QIODevice::WriteOnly | QIODevice::Append))
     if (fOut.open(QIODevice::WriteOnly | QIODevice::Append)) {
       QTextStream s(&fOut);
       //for (int i = 0; i < namesList.size(); ++i)
         s << name << '\n';
     //
     } else {
       std::cerr << "error opening output file\n";
       //return EXIT_FAILURE;
     }
     fOut.close();

}

static void readNames(){
    // read data



    if (fIn.open(QFile::ReadOnly | QFile::Text)) {
      QTextStream sIn(&fIn);

      while (!sIn.atEnd())
        //names += sIn.readLine();
        //cout << sIn.readLine().toStdString()<<endl;
       names.append(sIn.readLine());
    } else {
      std::cerr << "No registered names"<<endl;
      //return EXIT_FAILURE;
    }
}

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

static vector<Mat> loadImagesFromFolder(vector<Mat> images){

    //Mat vecOfLabels(1,numberOfPersons,CV_32SC1);
    string folderPath = "C:/BUILDS/FaceDetectionProject/myFaces/";
//                cout<<path<<endl;
    QDir dir(QString::fromStdString(folderPath));
    numberOfFolders = dir.count()-2;
    cout <<"Number of folders: " << numberOfFolders;

    for (int person = 1; person<=numberOfFolders; person++){
        string path = format("C:\\BUILDS\\FaceDetectionProject\\myFaces\\s%d",person);
        QDir dir(QString::fromStdString(path));
        int numberOfImages = dir.count()-2;

        for (int a=1; a<=numberOfImages;a++)  // a <=Count would do one too many...
        {
            string name = format("C:\\BUILDS\\FaceDetectionProject\\myFaces\\s%d\\%d.pgm",person,a);
            Mat img = imread(name,CV_LOAD_IMAGE_GRAYSCALE); // pgm implies grayscale, maybe even: imread(name,0); to return CV_8U
            if ( img.empty() )
            {
                cerr << "whaa " << name << " can't be loaded!" << endl;
                continue;
            }
            else{
                cout<<"loading :"<< name <<endl;
            }
            images.push_back(img);
//            //Mat newLabel(1,1,CV_32SC1);
//            //newLabel(1,1) = person;
//            vecOfLables(1,a) = person;
//            labels.push_back(person);

            // show result:
    //        cv::Mat testImage = images[a];
//            imshow("test",img);
                       // yes, you need the waitKey()
        }
    }
//    cout<< "images size: " << images.size()<<endl;
    return images;
}

static vector<int> loadLabels(vector<int> labels){

    for (int person = 1; person<=numberOfFolders; person++){
        string path = format("C:\\BUILDS\\FaceDetectionProject\\myFaces\\s%d",person);
        QDir dir(QString::fromStdString(path));
        int numberOfImages = dir.count()-2;
        for (int a=1; a<=numberOfImages;a++)  // a <=Count would do one too many...
        {
            labels.push_back(person);
        }
    }
    /*
    cout<< "labels size: " << labels.size()<<endl;*/
    return labels;
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    VideoCapture camera;

    //cv::VideoCapture camera("D:\\video.mp4");
   /* std::string videoStreamAddress = "rtsp://169.254.112.254/profile2/media.smp";

    if (!camera.open(videoStreamAddress)) {
        fprintf(stderr, "Error getting camera...\n");
        exit(1);
    }*/

    if (!camera.open(0)) {
        fprintf(stderr, "Error getting camera...\n");
        exit(1);
    }

    Ptr<FaceRecognizer> model1 = EigenFaceRecognizer::create();
    FaceClassifier classifier(model1);
    vector<Mat> images;             // Empty vector
    vector<qint32> labels;


    readNames();
    cout <<"Priniting names already registered : " <<names.size()<<endl;
     for(int i=0;i<names.size();i++){
         cout<<names[i].toStdString()<<endl;
     }


    //QFile fIn("names.txt");




    if (fullProgram){
    myTimer.start();
    int previousTime = 0;
    // Try opening camera

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO | cv::WINDOW_AUTOSIZE);

    VideoFaceDetector detector(CASCADE_FILE, camera);
    cv::Mat frame;
    int person = 0;
    int faceCount = 1;
    string userInput = "";
    double fps = 0, time_per_frame;
    while (true)
    {
        if(userInput== ""){
            cout << "Enter input: ";
            cin >> userInput;
            cout << userInput << endl;
            if(userInput=="s"){
                saveImages = true;
                //person++;
                string folderPath = "C:/BUILDS/FaceDetectionProject/myFaces/";
                //string path = format("C:/BUILDS/FaceDetectionProject/myFaces/s%d",person);///%d.pgm,faceCount
//                cout<<path<<endl;
                QDir dir(QString::fromStdString(folderPath));
                person = dir.count()-2 + 1;
                cout<<"Number of people updated : "<< person<<endl;

                cout<<"set SaveImges : True"<<endl;
                cout<<"Enter name of person"<<endl;
                string name;
                cin >> name;                
                cout << "Writing name to File and adding to list" << name;
                writeNameToFile(QString::fromStdString(name));  // write name to local text file
                names.append(QString::fromStdString(name));
            }
            if(userInput=="x"){
                saveImages = false;
                loadFromXml = true;
                cout<<"Loading Training data from XML";
            }
            if(userInput=="t"){
                trainImages = true;
                cout<<"Start Training Images" <<endl;
            }
            if(userInput=="p"){
                predict = true;
                cout<<"Start Recognition" <<endl;
            }
        }

        auto start = cv::getCPUTickCount();
        detector >> frame;
        auto end = cv::getCPUTickCount();

        time_per_frame = (end - start) / cv::getTickFrequency();
        fps = (15 * fps + (1 / time_per_frame)) / 16;


        if (detector.isFaceFound())
        {
            cv::rectangle(frame, detector.face(frame), cv::Scalar(255, 0, 0));

            int presentTime = myTimer.elapsed();
            int timeDelay = presentTime - previousTime;

            //saving images
            if(saveImages){
                if (faceCount<=maxImages && timeDelay>200) {


//                    if (!dir.exists()) { // Create folder
//                        dir.mkpath(QString::fromStdString(folderPath));
//                        cout<<"Creating myfaces folder"<<endl;
//                    }
                    string subFolderPath = format("C:/BUILDS/FaceDetectionProject/myFaces/s%d",person);///%d.pgm,faceCount
                    QDir dirSub(QString::fromStdString(subFolderPath));

                    if (!dirSub.exists()) {  // Create Subfolder
                        dirSub.mkpath(QString::fromStdString(subFolderPath));
                    }

                    string fileName = format("%d.pgm",faceCount);
                   QString imageName =  QString::fromStdString(subFolderPath) + "/" +  QString::fromStdString(fileName) ;
    //                imageName = path ;
                   cout<<imageName.toStdString()<<endl;
                    detector.saveFace(imageName);
                    previousTime = myTimer.elapsed();
                    faceCount++;
                }
                if(faceCount>maxImages)
                {
                    cout<< maxImages << " Images already saved" <<endl;
                    numberOfPersons++;
                    saveImages = false;
                    faceCount = 1;              // resest faceCount to 0 for new person
                    userInput = "";

                    /*for (int k=0;k<labels.size();k++){
                       cout<<labels[k]<<endl;
                    }*/
                }
            }
            if(loadFromXml)
            {

                QString fn_csv = "C:\\BUILDS\\FaceDetectionProject\\faceImagesCSV.csv";
              try {
                  readTrainingData_csv(fn_csv.toStdString(), images, labels);
                  cout<< "Number of images: " << images.size();
                  cout<< "Number of labels: " << labels.size();

              } catch (cv::Exception& e) {
                  cerr << "Error opening file \"" << fn_csv.toStdString() << "\". Reason: " << e.msg << endl;
                  // nothing more we can do
                  cout<<"CSV reading error";
                  qDebug()<<"CSV reading error";
                  exit(1);
              }
                userInput ="";

            }

            if(trainImages){

                    //load images & labels externally from Save Folder
                if(!loadFromXml){
                    cout<<"loading from folder";
                    images = loadImagesFromFolder(images);
                    labels = loadLabels(labels);
               }
                 loadFromXml = false;

                 testSample = images[images.size() - 1];
                 testLabel = labels[labels.size() - 1];
                 images.pop_back();
                 labels.pop_back();


                /*for (int k=0;k<labels.size();k++){
                   cout<<labels[k]<<endl;
                }*/

                    classifier.trainClassifier(images,labels);
                    trainImages = false;
                    userInput = "";
                    cout<<"Total Number of images: " << images.size()<< " Number of labels: " << labels.size()<<" Size of image sample " << images[2].size();
                    cout<<"Number of classes/folders/people: "<< numberOfFolders<< "Number of names in list: " << names.size()<<endl;
            }
            if(predict){
                cv::Mat unknownFace = detector.getFaceImage(); // return grey pgm size image
                 //cv::Mat unknownFace = cv::imread("C:/BUILDS/FaceDetectionProject/myFaces/s2/6.pgm",CV_LOAD_IMAGE_GRAYSCALE);

                //cout<<" Size of image: " << unknownFace.size <<endl;
               // int plabel = classifier.predictLabel(testSample);


                int predicted_label = -1;
                double predicted_confidence = 0.0;
                // Get the prediction and associated confidence from the model

                if(true){
                    predicted_label = classifier.predictLabel(unknownFace);
                    predicted_confidence = classifier.getConfidence(unknownFace,predicted_label);
                    QString predName = names.at(predicted_label-1);
                    cout<< "predicted, " << predicted_label <<"Name : " << predName.toStdString()<<endl <<" Confidence: " << predicted_confidence;

                }

            }
        //printf("Time per frame: %3.3f\tFPS: %3.3f\n", time_per_frame, fps);


            //cv::circle(frame, detector.facePosition(), 30, cv::Scalar(0, 255, 0));
        }

        cv::imshow(WINDOW_NAME, frame);
        if (cv::waitKey(25) == 27) break;
    }
    }   //end detection
    return a.exec();
}

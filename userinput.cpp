#include "userinput.h"
#include "VideoFaceDetector.h"

#include <QThread>
#include <stdio.h>
#include "iostream"
#include "istream"
#include "fstream"
#include <QDir>
#include <QTextStream>
#include <QDateTime>
#include <QThread>
#include <QDirIterator>


using namespace std;


int maxImages = 500;

int eigenFaces = 70; // 10 percent of total images

const cv::String    WINDOW_NAME("Camera video");
const cv::String    CASCADE_FILE = "C:/BUILDS/openCV/install/etc/haarcascades/haarcascade_frontalface_alt.xml";



UserInput::UserInput(QObject *parent): QThread(parent),m_classifier( EigenFaceRecognizer::create(eigenFaces))
//UserInput::UserInput(QObject *parent): QThread(parent),m_classifier( FisherFaceRecognizer::create(0,DBL_MAX))

{
   m_face = NULL;
   m_gotFrame = false;
   gotFace = false;
   save = false;
   train = false;
   load = false;
   predict = false;
   crossTest = false;
   saveVideo = false;

   m_action = "start";
   m_location = "a";  // To be set manually
   //testLocation = "b"; // Folder to be crosstested with

   //QStringList m_names;
    gotFaceCount = 0;       // count of face frames recieved
   person = 0;              // updated when saving images
   faceCount = 1;
   numberOfFolders = 0;     // updated when loading training images from folders. ie: one folder for each person/face.

   folderPath = "C:/BUILDS/FaceDetectionProject/WorkCongealed/" + string(m_location) + "/";

   nameList = "C:/BUILDS/FaceDetectionProject/WorkCongealed/names.txt";
   personHitVector = QVector<int>(1);
   ctpArray = QVector<int>(1);

  eyeCascadePath = "C:/BUILDS/openCV/install/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
   //eyeCascadePath = "C:/BUILDS/openCV/install/etc/haarcascades/parojos.xml";

   //setEyeCascade(eyeCascadePath);

//   Ptr<BasicFaceRecognizer> model1 = EigenFaceRecognizer::create(eigenFaces);
//   FaceClassifier m_classifier(model1);



}

//void UserInput::setEyeCascade(const std::string eyeCascadeFilePath)
//{
//    if (eyeCascade == NULL) {
//        eyeCascade = new cv::CascadeClassifier(eyeCascadeFilePath);
//        cout<<"Eye cascade loaded"<< endl;
//    }
//    else {
//         eyeCascade->load(eyeCascadeFilePath);
//    }

//    if ( eyeCascade->empty()) {
//        cout << "Error creating cascade classifier. Make sure the file" << std::endl
//            << eyeCascadeFilePath << " exists." << std::endl;
//    }
//}

void UserInput::writeNameToFile(QString name){
     QFile fNames(nameList);
     if (fNames.open(QIODevice::WriteOnly | QIODevice::Append)) {
       QTextStream s(&fNames);
       //for (int i = 0; i < namesList.size(); ++i)
         //s << "\n" <<name ;
       s << name <<"\n" ;
     //
     } else {
       std::cerr << "error opening output file\n";
       //return EXIT_FAILURE;
     }
     fNames.close();
}

void UserInput::readNames(){
    QFile fNames(nameList);
    if (fNames.open(QFile::ReadOnly | QFile::Text)) {
      QTextStream sIn(&fNames);
      while (!sIn.atEnd())
        m_names.append(sIn.readLine());
    } else {
      std::cerr << "No registered names"<<endl;
      //return EXIT_FAILURE;
    }
    cout <<"Priniting names already registered : " <<m_names.size()<<endl;

}

void UserInput::printNames(){
     for(int i=0;i<m_names.size();i++){
          cout<<m_names[i].toStdString()<<endl;

     }
}


//bool UserInput::eyesFound(Mat faceImage){
//    std::vector<cv::Rect> m_allEyes;
//    bool eyes = false;
//   eyeCascade->detectMultiScale(faceImage, m_allEyes, 1.1, 0, 0,
//        cv::Size(faceImage.rows /20, faceImage.cols/ 20),
//        cv::Size(faceImage.rows * 1 /2, faceImage.cols * 1/2));
//   int length = static_cast<int>(m_allEyes.size());
//   int centerx = ceil(faceImage.cols/2);
//   if(length>=2){
//       int sumx = 0;
//       for(int i=0;i>length;i++){
//           if(m_allEyes.at(i).x>centerx)
//            sumx = sumx +1;
//           else
//               sumx = sumx-1;
//       }
//       if(sumx<length){// both eyes found
//            eyes = true;
//       }
//       cout<<length<<"Eyes found"<<endl;
//   }
//   return eyes;

//}

void UserInput::alignFace(Mat faceImage,int s){
    bool eyes = false;
    std::vector<cv::Rect> m_allEyes;
    QString imageName = "C:/BUILDS/FaceDetectionProject/Work/Alignment/eyes" + QString::fromStdString(to_string(s)) + ".png";
    QString alignedImage = "C:/BUILDS/FaceDetectionProject/Work/Alignment/alignedFace"  + QString::fromStdString(to_string(s)) + ".png";
//   eyeCascade->detectMultiScale(faceImage, m_allEyes, 1.1, 0, 0,
//        cv::Size(faceImage.rows /20, faceImage.cols/ 20),
//        cv::Size(faceImage.rows * 1 /4, faceImage.cols * 1/4));

   int length = static_cast<int>(m_allEyes.size());
   int centerFace = ceil(faceImage.cols/2);
   std::vector<cv::Point> rectPoints;
   //std::vector<cv::Rect> bestRects;
   Point P1,P2;
   Rect rectL, rectR;
   if(length>=2){
       int sumx = 0;
       int xmin = faceImage.cols;
       int xmax = 0;
       int l = 0;
       int r = 0;
       for(int i=0;i<length;i++){
           Rect eyeRect = m_allEyes.at(i);
           Point center;

           center.x = ceil(eyeRect.x + eyeRect.width/2);
           center.y = ceil(eyeRect.y + eyeRect.height/2);
           //cout<<"Rect " << i << ":" << eyeRect.x + eyeRect.width/2 << "; " <<eyeRect.y  + eyeRect.height/2<< ";" << eyeRect.width << ";" << eyeRect.height <<endl;
           cout<<center.x <<center.y<<endl;
           rectPoints.push_back(center);

           if(center.x>centerFace){
            sumx = sumx +1;
           }
           else{
               sumx = sumx-1;
           }
           if(center.x<xmin){
               xmin = center.x;
               P1 = rectPoints.at(i);
               l = i;
           }
           if(center.x>xmax){
               xmax = center.x;
               P2 = rectPoints.at(i);
               r = i;
           }

       }

       rectL = m_allEyes.at(l);  // left eye
       rectR = m_allEyes.at(r);  // right eye
       if(abs(sumx)!=length){   // both eyes found
            eyes = true;
       }
      // cout<<length<<"Eyes found "<< "Geo Sum:" << sumx << endl;
   }
   if(eyes){ // save image with detected eyes

       cv::rectangle(faceImage,rectL,cv::Scalar(255,0,0));
       cv::rectangle(faceImage,rectR,cv::Scalar(255,0,0));
      // cout<<length<<"Eyes found "<< "Points: P1[" << P1.x << ","<< P1.y << "] P2: [" <<P2.x << "," << P2.y<<"]"<<endl;
       //cv::imwrite(imageName.toStdString(),faceImage);

       // rotate image
       double teta = atan2(double(P2.y-P1.y),double(P2.x-P1.x)) ;
       teta = teta*180/3.142;
       cv::Point2f pc(faceImage.cols/2.,faceImage.rows/2.);
       cv::Mat r = cv::getRotationMatrix2D(pc, teta, 1.0);

       cv::warpAffine(faceImage,faceImage, r, faceImage.size()); // what size I should use?

       cv::imwrite(alignedImage.toStdString(), faceImage);

       cout<<"Angle is : " << teta <<endl;
   }


//   if(!m_allEyes.empty()){
//    cout<<length<<"found" <<endl;
//    cout<< m_allEyes.at(0).x << m_allEyes.at(0).y<<endl;
//    if(length>=2){
//        cv::imwrite(imageName.toStdString(),faceImage);
//    }

//   }
//   else
//       cout<<"Eyes not found" <<endl;

    // Locate eye coordinates
    //Find angle. Rotate image
    //Find distance scale image
    //Find offset, translate image

}

/*Load Images from folder with given fpath*/
void UserInput::loadImagesFromFolder(string fpath, int num){
    int imageCount = 1;
    QDir dir(QString::fromStdString(fpath));

    numberOfFolders = dir.count()-2;
    cout <<"Number of folders: " << numberOfFolders;
    QStringList folders = dir.entryList();
    for (int folder = 2; folder<folders.length(); folder++){ // For each folder
        cout<<"folderpath:" <<fpath<<endl;
         string subfolderpath = fpath + folders.at(folder).toStdString();
        QDir subdir(QString::fromStdString(subfolderpath));
        int numberOfImages = subdir.count()-2;
        if(num!=0){
            numberOfImages = num;
             }


          cout<<"Images in "<<subfolderpath<< " is " << numberOfImages<<endl;
          QStringList images = subdir.entryList();
          cout<<"list size" <<images.length()<<endl;
        for (int a=2; a<numberOfImages+2;a++) { // load Images in Folder, ///a should be 2 entryList lists n of folders + 2

            //string name = subfolderpath + "/" + std::to_string(a) + ".pgm";
            string name = subfolderpath + "/" +   images.at(a).toStdString();

            Mat img = imread(name,CV_LOAD_IMAGE_GRAYSCALE); // pgm implies grayscale, maybe even: imread(name,0); to return CV_8U
            cv::equalizeHist(img,img); // hist equalization
            if ( img.empty() )
            {
                cerr << "ERROR: " << name << " can't be loaded!" << endl;
                continue;
            }
            else{
                cout<<"loading :"<< name << " image count " << imageCount++<< endl;
            }
            m_images.push_back(img);
        }
    }

}


void UserInput::assignLabels(string fpath){

    QDir dir(QString::fromStdString(fpath));
    numberOfFolders = dir.count()-2;
    cout <<"Number of folders: " << numberOfFolders;

    for (int person = 1; person<=numberOfFolders; person++){ // For each folder
        cout<<"folderpath:" <<fpath<<endl;
        string subfolderpath = fpath +"s" + std::to_string(person);
        QDir dir(QString::fromStdString(subfolderpath));
        int numberOfImages = dir.count()-2;

         cout<<"Labels in "<<subfolderpath<< "is" << numberOfImages<<endl;
        for (int a=1; a<=numberOfImages;a++)  // a <=Count would do one too many...
        {
            m_labels.push_back(person);
        }
    }

}

void UserInput::readLabels(string fpath,int num){

    QDir dir(QString::fromStdString(fpath));
    numberOfFolders = dir.count()-2;
    cout <<"Number of folders in Test folder: " << numberOfFolders <<endl;
    QStringList folders = dir.entryList();
   cout<< "Number of entries in Qlist" << folders.length()<<endl;

    for (int folder = 2; folder<folders.length(); folder++){
       // cout<<"Folder" << folder<<endl;
        QString label = folders.at(folder).right(1);

        person = label.toInt();  // label
        string subfolderpath = fpath + folders.at(folder).toStdString();
        QDir subdir(QString::fromStdString(subfolderpath));
        int numberOfImages = subdir.count()-2;
        if(num!=0){
            numberOfImages = num;
             }


        for (int a=0; a<numberOfImages;a++)  // a <=Count would do one too many...
        {
           m_labels.push_back(person);
        }
        cout<< "Person: "<< label.toStdString()<< " Labels: " << numberOfImages<< endl;
    }
    cout<<"Reading Labels finished, total: " << m_labels.size()<<endl;

    //return m_labels;
}


void UserInput::run(){
    while(true){


        if(m_action=="start"){
            UserInput::readNames();
            UserInput::printNames();
            cout<<"Location folder : "<<m_location<<endl;
            getUserInput();
        }       
        if(m_action==""){
            //UserInput::printNames();
            getUserInput();
        }
        if(save){          
            if(gotFace)
                saveImages();
            else
                cout<<"Did not receive face from Face Detector"<<endl;
        }
        if(train){
            int num;
            cout<<"Enter number of images per person: " ;
            cin>>num;
            cout<<""<<endl;
            trainImages(num);
        }
        if(load){
            loadModel();
        }
        if(predict){
            //emit requestFace();
            if(gotFace){
                gotFaceCount++;
                int label = predictImages();
            }

//            else
//                cout<<"Looking for face from Face Detector" <<endl;
        }
        if(crossTest){
            runCrossTest();
        }
        cv::Mat face = getCurrentFace();

        cv::imwrite("savedFace.jpg",m_face);



    }
}
/* Slot to catch face from FaceDetector signal sendFace() and assign to m_face.
*/
void UserInput::catchFace(cv::Mat pFace){
    if(!pFace.empty()){
        gotFace = true;
        m_face = pFace;
    }
    else{
        gotFace = false;
       // cout<<"Did not catch face"<<endl;
    }

    //cout<<"Catching frame" <<endl;


}

cv::Mat UserInput::getCurrentFace(){
    //if(frameReady())

         return m_face;
}

void UserInput::getUserInput(){
    cout << "Enter input: s-save images,a-align saved images, t-train, p-predict, c-crosstest : ";
    string action;
    cin >> action;
    m_action = QString::fromStdString(action);
    if(m_action=="s"){ // setup folder
        saveVideo = true;
        emit savingFacesMode(saveVideo);

        cout<<"Enter location : a/b/c" ;
        cin >> m_location;
        cout<<"Location set to: " << folderPath <<endl;
         //string folderPath = "C:/BUILDS/FaceDetectionProject/Work/" + string(m_location) + "/";
        QDir dir(QString::fromStdString(folderPath));
        cout<<"Number of people: " << dir.count()-2;
        cout<<"Enter name of person: ";
        string name;
        cin >> name;
        person = dir.count()-2 +1; // dir.Count() returns number of files + 2. Subtract names.txt file
        cout<<"Number of people updated : "<< person<<" set SaveImges : True; "<<" Writing name to File and adding to list: " << name;

        UserInput::writeNameToFile(QString::fromStdString(name));  // write name to local text file
        m_names.append(QString::fromStdString(name));

        save = true;
    }
    if(m_action=="a"){
        //congealImages();
        QString imageListFile = "pFacesIn.txt";
        QString gDirectory = "C:/BUILDS/FaceDetectionProject/WorkCongealed/Temp";
        bool con = true;
        emit congealImages(imageListFile,gDirectory,con);


    }
    if(m_action=="t"){
        cout<<"Start Training Images from" <<m_location<<endl;
        train = true;
    }
    if(m_action=="l"){
        cout<<"Load Trained Model"<<endl;
        load = true;
    }
    if(m_action=="p"){
        // emit savingFacesMode(true);
        QDir dir(QString::fromStdString(folderPath));
        numberOfFolders  = dir.count()-2;
        personHitVector.resize(numberOfFolders ); // extend predictor measure array to number of updated people

        predict = true;
        cout<<"Start Recognition: Number of people registered: "<< numberOfFolders << endl;
    }
    if(m_action=="c"){
        cout<<"Trained location is " << m_location << ". Enter test location:" ;
        cin >> testLocation;
        cout << "" <<endl;
        crossTestFolder = "C:/BUILDS/FaceDetectionProject/WorkCongealed/" + string(testLocation) + "/";
        crossTest = true;
       // cout<<"Cross Test with "<< crossTestFolder <<endl;
    }
}

void UserInput::saveImages(){

    if (faceCount<=maxImages) {

         string subFolderPath = folderPath +"s" + std::to_string(person);
        QDir dirSub(QString::fromStdString(subFolderPath));
        if (!dirSub.exists()) {  // Create Subfolder
            dirSub.mkpath(QString::fromStdString(subFolderPath));      }
        string fileName = format("%d.pgm",faceCount);
        QString imageName =  QString::fromStdString(subFolderPath) + "/" +  QString::fromStdString(fileName) ;

         cv::Mat faceImageGrayResized = getCurrentFace();
       // bool eyes = eyesFound(faceImageGrayResized);
         bool eyes = true;
        if(eyes){ // eyes found
            cv::imwrite(imageName.toStdString(),faceImageGrayResized);
            cout<<"saving Face Image"<< imageName.toStdString()<<endl;
            faceCount++;
        }
        else
            cout<<"Eyes not found. Not saving Image"<<endl;
       // previousTime = myTimer.elapsed();

    }
    if(faceCount>maxImages)
    {
        cout<< maxImages << " Images already saved" <<endl;
       // numberOfPersons++;
        save = false;
        saveVideo = false;
        faceCount = 1;              // resest faceCount to 0 for new person
        m_action = "";
        emit savingFacesMode(saveVideo);        // end saving video clip
    }

}

//void UserInput::congealImages(string inputFile,string path,bool congSignal)
//{
////    cout<<"Start Congealing" <<endl;
////    m_congealer.congeal();
////    cout<<"Congealing finished"<<endl;
////    m_action="";

//}

void UserInput::createClassifier(){
//    Ptr<BasicFaceRecognizer> model1 = EigenFaceRecognizer::create(eigenFaces);
//    FaceClassifier classifier(model1);
}

void UserInput::trainImages(int num){
    // load Images & labels from folderPath
    QTime myTimer;
    myTimer.start();
    UserInput::loadImagesFromFolder(folderPath,num);
    UserInput::readLabels(folderPath,num);
    //UserInput::assignLabels(folderPath);
    cout<<"Total images: " <<m_images.size()<<endl;

//    for (int i=0;i<m_images.size();i++){
//        Mat image1 = m_images.at(i);
//        alignFace(image1,i);
//    }


    m_classifier.trainClassifier(m_images,m_labels);

    int trainingTime = myTimer.elapsed();
    m_classifier.saveModel();
    cout<<"Training Completed. Elapsed time : " << trainingTime/1000 <<endl;
    cout<<"Total Number of images: " << m_images.size()<< " Number of labels: " << m_labels.size()<<" Size of image sample " << m_images[2].size();
    cout<<"Number of classes/folders/people: "<< numberOfFolders<< "Number of names in list: " << m_names.size()<<endl;
    for(int i = 0;i<m_names.size();i++){
    cout<<"Names in list are : " << m_names.at(i).toStdString()<<endl;             }

    cv::Mat vectors = m_classifier.getEigenVectors();
    cv::Mat values = m_classifier.getEigenValues();
    cout<<"Size of EigenVectors " << vectors.size;
    cout<<"Size of EigenValues " << values.size;



    m_images.clear();   // clear vectors
    m_labels.clear();
    train = false;
    m_action = "";


}

int UserInput::predictImages(){
    cv::Mat unknownFace = getCurrentFace(); // return grey pgm size image

    int label = -1;
    double err = 0;
    // Get the prediction and associated confidence from the model


    if(!unknownFace.empty()){
        label = m_classifier.predictLabel(unknownFace);
        double err = m_classifier.getProjectedDifference(unknownFace);
        QString predName = m_names.at(label-1);

        personHitVector[label-1] = personHitVector.at(label-1)+1;
        double max = 0;
        int maxPerson;
        int rateCount = gotFaceCount%100;
       // cout<<rateCount<<" :";
        if(rateCount==99){
            for(int i=0;i<personHitVector.size();i++){
                double rate = personHitVector.at(i)*100/rateCount;
                if(rate>max){
                    max = rate;
                    maxPerson = i;
                }
                cout<<rate<<"; " ;
            }
            cout<<m_names.at(maxPerson).toStdString()<<" , ";
        }

        if(rateCount==0)
        {
             cout<<"Refresh hit rate" << endl;
             for(int j=0;j<personHitVector.size();j++){
                     personHitVector[j]=0;
             }
        }

        outConfidence(err,0,label);

        /*

        //QString predName = m_names.at(maxPerson);
        //cout<<"       "<< predName.toStdString() <<" confidence: " << err <<endl;
        QFile outFile("C:/BUILDS/FaceDetectionProject/Work/projectedDifferences.txt");

        if (outFile.open(QIODevice::WriteOnly | QIODevice::Append)) {
          QTextStream s(&outFile);
            s <<QDateTime::currentDateTime().toString()<<'\t'<< err << '\n';
        } else {
          std::cerr << "error opening output file\n";           }
        outFile.close(); */
    }
    else{
        cout<<"Captured image empty" << endl;
    }
    return label;
}

void UserInput::outConfidence(double err,int o_label, int p_label){
    QFile outFile("C:/BUILDS/FaceDetectionProject/Work/projectedDifferences.txt");

    if (outFile.open(QIODevice::WriteOnly | QIODevice::Append)) {
      QTextStream s(&outFile);
        s <<QDateTime::currentDateTime().toString()<<'\t'<< o_label << "\t" << p_label <<"\t" << err << '\n';
    } else {
      std::cerr << "error opening output file\n";           }
    outFile.close();
}

void UserInput::output(){

    QFile outputFile("C:/BUILDS/FaceDetectionProject/WorkCongealed/output.txt");
    QString out = "";
    for(int i=0;i<ctpArray.length();i++){
        out = out + QString::fromStdString(std::to_string( ctpArray[i])) + " " ;
    }
    if (outputFile.open(QIODevice::WriteOnly | QIODevice::Append)) {
      QTextStream s(&outputFile);
        s <<QDateTime::currentDateTime().toString()<<'\t'<< QString::fromStdString(m_location) << "\t" << QString::fromStdString(testLocation) <<"\t" << out << '\n';
    } else {
      std::cerr << "error opening output file\n";
    }
    outFile.close();
}

void UserInput::loadModel(){
    m_classifier.loadModel();
    cv::Mat vectors = m_classifier.getEigenVectors();
    cv::Mat values = m_classifier.getEigenValues();
    cout<<"Size of EigenVectors " << vectors.size;
    cout<<"Size of EigenValues " << values.size;

    load = false;
    m_action = "";

}

void UserInput::runCrossTest(){

   UserInput::loadImagesFromFolder(crossTestFolder,0);
   UserInput::readLabels(crossTestFolder,0);
   cout<<"Number of test images: "<< m_images.size();
   cout<<" Number of test labels: " << m_labels.size();
   QDir dir(QString::fromStdString(crossTestFolder));
   int folders = dir.count()-2;

   if(m_images.size()==m_labels.size()){
       int tp = 0; // true positives
       int ctp = 0; // class true positives

       QVector<int> cLabels = QVector<int>(1);                              //class count array
       cout<<"Number of test people/folders "<<folders<<endl;
       int NP = m_names.length();
       ctpArray.resize(NP);
       cLabels.resize(NP);
      //int c = 0;//class number
       int n = 1;                                                           // class labels
       int o_labelPrevious = 1;                                             // first class label is 1
       for(int i=0;i<m_images.size();i++){
          cv::Mat unknownFace = m_images.at(i);
          double err = m_classifier.getProjectedDifference(unknownFace);

          int p_label = m_classifier.predictLabel(unknownFace);             // predicted label
          int o_label = m_labels.at(i);                                     // original label
          cLabels[o_label-1]= cLabels[o_label-1] +1;                        // increment class count
          if(i>0)
              o_labelPrevious = m_labels.at(i-1);
          if(o_label!=o_labelPrevious){
              //cout<<"class:"<<c<< "postives "<< ctp << "Total class labels: " << n<<endl;
              cout<<"Class changed"<<endl;
              n=1;
              ctp = 0;
          }
          else{
              n++;

          }
          if (p_label==m_labels.at(i)){                                     //if true positive
              tp++;                                                         //increase total tp count
              ctp++;                                                        //increase class tp count
              ctpArray[m_labels.at(i)-1] = ctp;
          }
          cout << i+1 << ": Actual : " << o_label<<";  Predicted:  " << p_label << " positives: " << ctp << " of " << n <<endl;
          outConfidence(err,o_label,p_label);
      }
      int total = m_images.size();
      int rate = ceil(tp*100/total);
      cout<<"Finished Cross Test. True Positives " << tp <<"/" << m_images.size() << ":  " << rate<<endl;
      cout<<"Class Results : [ " <<endl;
      for(int i=0;i<ctpArray.length();i++){
          cout<< ctpArray[i] <<"/" << cLabels[i] <<endl;
      }

   }
   output(); //printoutput
   m_images.clear();   // clear vectors
   m_labels.clear();
   ctpArray.clear();
   crossTest = false;
   m_action = "";

}


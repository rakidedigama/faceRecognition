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

   folderPath = "C:/BUILDS/FaceDetectionProject/Work2/" + string(m_location) + "/";

   nameList = "C:/BUILDS/FaceDetectionProject/Work2/names.txt";
   personHitVector = QVector<int>(1);
   ctpArray = QVector<int>(1);


//   Ptr<BasicFaceRecognizer> model1 = EigenFaceRecognizer::create(eigenFaces);
//   FaceClassifier m_classifier(model1);



}

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
    cout << "Enter input: s-save images, t-train, p-predict : ";
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
        crossTestFolder = "C:/BUILDS/FaceDetectionProject/Work2/" + string(testLocation) + "/";
        crossTest = true;
       // cout<<"Cross Test with "<< crossTestFolder <<endl;
    }
}

void UserInput::saveImages(){

    if (faceCount<=maxImages) {
        // string folderPath = "C:/BUILDS/FaceDetectionProject/Work/" + string(m_location) + "/";

         string subFolderPath = folderPath +"s" + std::to_string(person);
        QDir dirSub(QString::fromStdString(subFolderPath));
        if (!dirSub.exists()) {  // Create Subfolder
            dirSub.mkpath(QString::fromStdString(subFolderPath));      }
        string fileName = format("%d.pgm",faceCount);
        QString imageName =  QString::fromStdString(subFolderPath) + "/" +  QString::fromStdString(fileName) ;

         cv::Mat faceImageGrayResized = getCurrentFace();

        cv::imwrite(imageName.toStdString(),faceImageGrayResized);
        cout<<"saving Face Image"<< imageName.toStdString()<<endl;

       // previousTime = myTimer.elapsed();
        faceCount++;
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

    QFile outputFile("C:/BUILDS/FaceDetectionProject/Work2/output.txt");
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


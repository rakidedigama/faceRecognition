#ifndef CONGEALER_H
#define CONGEALER_H

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "math.h"
#include "float.h"
#include <QDir>

#include <QThread>

using namespace std;
using namespace cv;

class Congealer: public QThread
{
    Q_OBJECT
public:
    explicit Congealer(QObject *parent = 0);
    void congeal();
    void makeEntFrame(vector<vector<float> > &logDistField, vector<pair<int, int> > &randLocs, char *fn, float minEnt, float maxEnt,
              int innerDimH, int innerDimW, int paddingH, int paddingW);
    void MakeFrame(vector<IplImage *> &images, vector<vector<float> > &v, int h, int w,
               vector<vector<vector<float> > > &bSq, string basefn, int frameIndex, int maxFrameIndex);
    void showResults(char *imageListFn, vector<vector<float> > &v, int h, int w,
             int paddingH, int paddingW, vector<pair<int,float> > &indexProbPairs,
             bool display, string dDirectory, bool generateFinal, string gDirectory);
    float computeLogLikelihood(vector<vector<float> > &distField, vector<vector<float> > &fids, int numFeatureClusters);
    float computeEntropy(vector<vector<float> > &distField, int numFeatureClusters);
    void getNewFeatsInvT(vector<vector<float> > &newFIDs, vector<vector<vector<float> > > &originalFeats,
                 vector<float> &vparams, float centerX, float centerY, vector<pair<int, int> > &randLocs);
    float dist(vector<float> &a, vector<float> &b);
    void computeGaussian(vector<vector<float> > &Gaussian, int windowSize);
    void getSIFTdescripter(vector<float> &descripter, vector<vector<float> > &m, vector<vector<float> > &theta, int x, int y, int windowSize, int histDim, int bucketsDim,
                   vector<vector<float> > &Gaussian);
    float findPrincipalAngle(float a1, float v1, float a2, float v2, float a3, float v3);
    void setRandLocs(vector<pair<int, int> > &randLocs, int h, int w, int paddingH, int paddingW, bool nonRand);
    void reorientM(vector<vector<float> > &m, vector<vector<float> > &newM,
               vector<vector<float> > &theta, vector<vector<float> > &newTheta,
               float angle, float cx, float cy, int windowSize);
protected:
    virtual void run();


public slots:
   void startCongeal(QString inputFile, QString path, bool congSignal);

private:

    vector<IplImage *> baseImages;
    vector<IplImage *> originalImages;
    vector<vector<float> > cropParams;
    vector<int> imageIndex;
    string imageFn;

    bool animation ;
    bool outputParams;
      bool visualize;
 bool generateFinal ;
 bool verbose;
 bool nonRand;
 bool display;
 bool align;
 string m_inputFile;
 string m_path;

};

#endif // CONGEALER_H

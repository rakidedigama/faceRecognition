#-------------------------------------------------
#
# Project created by QtCreator 2017-08-15T15:21:55
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = FaceDetectionApp1
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

HEADERS  += VideoFaceDetector.h \
    faceclassifier.h \
    dsagdsgf.h

SOURCES += main.cpp\
         VideoFaceDetector.cpp \
    faceclassifier.cpp

INCLUDEPATH += C:\BUILDS\openCV\install\include

LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_core330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_highgui330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_imgcodecs330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_imgproc330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_features2d330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_calib3d330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_video330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_videoio330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_videostab330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_objdetect330d.lib
LIBS += C:\BUILDS\openCV\install\x64\vc11\lib\opencv_face330d.lib

# faceRecognition

This Project uses the OpenCV FaceRecognizer class along with Haar Cascade classifier for Face recongition and detection, respectively. 
The FaceClassifier class abstracts the BasicFaceRecognizer class with the ability to change type of classifier (based on features- eigenface or fisher-faces). 

The project consists of four processes running in dedicated QThreads. 
  1.  The CameraViewer class grabs frames from a specified camera and passed it on to the FaceDetector class. 
  2.  The FaceDetector class detects a new face, or tracks an already detected face using HAAR cascades (and template matching technique when HAAR Cascade fails). If a face is detected, it is cropped from the background of the original frame and passed on to the UserInput class. 
  3.  The UserInput class is the main program that interacts with the user in this console application. It allows the user to chose any of the following options. 
      * Saving Images - S
      * Aligning saved images (using Congealing) - C
      * Training images - T
      * Live prediction from camera feed (using a voting method for 100 frames) - P
      * Offline crosstesting platform for previously saved images - C
      
   --- Project Directories are defined in the source code. 
      
  4. The Congealing process is also implemented on a seperate thread (Congealer class) since it is computationally heavy. ** Input parameters are defined locally at the moment. 
      

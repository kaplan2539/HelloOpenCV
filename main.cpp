#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <string>

using namespace cv;

void detectAndDisplay( Mat frame );

String face_cascade_name =
  "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
std::string window_name = "Hello OpenCV";

std::string text = "initializing...";
int fontFace = FONT_HERSHEY_PLAIN;
double fontScale = 1;
int thickness = 2;

int frameWidth  = 160;
int frameHeight = 120;

int main( int argc, const char** argv )
{
  CvCapture* capture;
  Mat frame;

  if( !face_cascade.load( face_cascade_name ) ){
    std::cerr << "ERROR: could not load "<< face_cascade_name << "\n";
    return -1;
  };

  capture = cvCaptureFromCAM( -1 );
  if( capture )
  {
    cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,frameWidth);
    cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,frameHeight);
    while( true )
    {
      frame = cvQueryFrame( capture );

      if( !frame.empty() ) {
        detectAndDisplay( frame );
      } else {
        std::cerr<< "ERROR: empty frame!\n"; break;
      }

      int c = waitKey(1);
      if( (char)c == 'c' ) {
        break;
      }
    }
  }
  return 0;
}

void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  static int n_frames=0;
  static int frame_probe=30;
  static double last_ticks = getTickCount();

  if(n_frames >= frame_probe) {
  	double fps = n_frames / (getTickCount() - last_ticks) * getTickFrequency();
  	std::stringstream ss;
  	ss << fps << " fps";
  	text = ss.str();
	n_frames=0;
	last_ticks = getTickCount();
  }

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );

  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2,
                                 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5,
                  faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5),
             0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;
  }
  rectangle(frame,Point(0,frameHeight-21),Point(frameWidth-1,frameHeight-1),
            Scalar::all(0),CV_FILLED);
  putText(frame, text, Point(5,frameHeight-2), fontFace, fontScale,
        Scalar::all(255), thickness, 8);

  imshow( window_name, frame );

  n_frames++;
}

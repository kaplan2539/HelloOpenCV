#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int frameWidth=320;
int frameHeight=240;

int showHist(Mat &src)
{
/// Separate the image in 3 places ( B, G and R )
  vector<Mat> bgr_planes;
  split( src, bgr_planes );

  /// Establish the number of bins
  int histSize = 256; 
  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );
}

void showSingle(Mat &what)
{
  Mat hist;

  /// Establish the number of bins
  int histSize = 256; 
  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  calcHist( &what, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 1, NORM_MINMAX );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  namedWindow("single", CV_WINDOW_AUTOSIZE );
  imshow("single", histImage );
}

void calcCMS(Mat& I, int thresh,int &x, int &y)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    int channels = I.channels();

    int nRows = I.rows;
    int nCols = I.cols * channels;

    int n_bins = 10;
    int binw_x = I.cols / n_bins;
    int binw_y = I.rows / n_bins;

    int i,j;
    uchar* p;
    
    float cms_x=0.0, cms_y=0.0, denom=0.0;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            if( p[j] > thresh ) {
              cms_x += j*p[j];
              cms_y += i*p[j];
              denom += p[j];
            }
        }
    }

    x = cms_x / denom;
    y = cms_y / denom;

}

void rebin(Mat &I, Mat &O, int binsize=20, int thresh=0)
{
    int channels = I.channels();

    int iRows = I.rows;
    int iCols = I.cols * channels;

    int oRows = iRows / binsize;
    int oCols = iCols / binsize;

    O = Mat::zeros(oRows,oCols,CV_32F);

    uchar *ip,*op;
    for( int i = 0; i < iRows; ++i)
    {
        ip = I.ptr<uchar>(i);
        op = O.ptr<uchar>( i/binsize );
        for ( int j = 0; j < iCols; ++j)
        {
            if( ip[j] > thresh ) {
              op[ j / binsize ] += ip[j];
            }
        }
    }
    normalize(O, O, 1.0, 0, NORM_MINMAX, -1, Mat() );
}

int main( int, char** argv )
{
  CvCapture* capture;
  Mat src, dst, diff;

  capture = cvCaptureFromCAM( -1 );

  // Set up the detector with default parameters.
  SimpleBlobDetector detector;
 
  // Detect blobs.
  std::vector<KeyPoint> keypoints;

  if( capture )
  {
    cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,frameWidth);
    cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,frameHeight);

    while( true )
    {
      src = cvQueryFrame( capture );
      if( src.empty() )
      { return -1; }

      /// Separate the image in 3 places ( B, G and R )
      vector<Mat> bgr_planes;
      split( src, bgr_planes );

      bool uniform = true; bool accumulate = false;

      diff=bgr_planes[2]-bgr_planes[0]-bgr_planes[1];

      int x,y;
      calcCMS(diff,20,x,y);
      circle(src, Point(x,y), 10, Scalar(255,255,255));

//      imshow("blue", bgr_planes[0] );
//      imshow("green", bgr_planes[1] );
//      imshow("red", bgr_planes[2] );
//      imshow("diff", diff );

      Mat small,scaled,normed;
    
      normalize(diff, diff, 255, 0, NORM_MINMAX, -1, Mat() );

//      rebin(diff,small);
      imshow("src", src );

//      detector.detect( small, keypoints);

//      resize(small,scaled,diff.size());
//      imshow("small", scaled );

//      for(int i=0; i<keypoints.size(); i++) {
//          KeyPoint kp = keypoints[i];
//          printf( "%d %d\n", kp.pt.x, kp.pt.y );
//      }


//      showHist(src);
//      showSingle(diff);
      int c = waitKey(1);
      if( (char)c == 'c' ) {
        break;
      }
    }
  }

  return 0;

}

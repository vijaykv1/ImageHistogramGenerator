#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Threshold.h"
#include "Histogram.h"
#include "PointOperations.h"
#include "Timer.h"


int main(int argc, char *argv[])
{
    //read image
    cv::Mat img = cv::imread("../data/lena.tiff"); // reading the image ... 
    //display image
    //cv::imshow("Image", img);
    
    //convert to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, CV_BGR2GRAY); // convert to grey scale image ...
    //display grayscale
    cv::imshow("Grayscale", imgGray); 

    //declare output variables
    /* Image outputs */
    cv::Mat imgThresholded; // for the thresholded image 
    cv::Mat imgAdjustedContrast; // for the contrast adjusted image 
    cv::Mat imgAdjustedBrightness; // for the brightness adjusted image 
    cv::Mat imgInverted; // for the inverted image 
    cv::Mat imgQuantized; // for the quantized image 

    /* Histogram lists */
    cv::Mat histGray; // histogram of the grey image 
    cv::Mat histThresholded; // histogram of the thresholded image 
    cv::Mat histAdjustedContrast; // histogram of the contrasted image 
    cv::Mat histAdjustedBrightness; // histogram of the brightness adjusted image 
    cv::Mat histInverted; // histogram of the inverted image 
    cv::Mat histQuantized; // histogram of the quantized image 

    uchar min, max, mean;

    //create class instances
    Threshold *threshold = new Threshold();
    Histogram *histogram = new Histogram();
    PointOperations *pointOperations = new PointOperations();
    
    // begin processing ///////////////////////////////////////////////////////////
    
    
    // calculate and display histogram of grayscale image
    histogram->calcHist(imgGray, histGray);
    histogram->show("Histogram Grayscale", histGray);

    // calculate and display histogram of grayscale image with OpenCV function
    histogram->calcHist_cv(imgGray, histGray);
    histogram->show("histogram_grayscale_cv", histGray); 

    // threshold and display image
    threshold->loop_ptr2(imgGray, imgThresholded, 50);
    cv::imshow("Thresholded", imgThresholded);

    // calculate and display histogram of thresholded image
    histogram->calcHist(imgThresholded, histThresholded);
    histogram->show("Histogram Thresholded", histThresholded);

    // calculate minimum, maximum and mean
    histogram->calcStats(histGray, min, max, mean);
    // print results
    std::cout << "Min:  " << (unsigned)min << std::endl;
    std::cout << "Max:  " << (unsigned)max << std::endl;
    std::cout << "Mean: " << (unsigned)mean << std::endl; 

    // adjust the image's contrast and display result
    pointOperations->adjustContrast(imgGray, imgAdjustedContrast, 0.5, 127);
    cv::imshow("Adjusted Contrast", imgAdjustedContrast);
    // calculate and show histogram
    histogram->calcHist(imgAdjustedContrast, histAdjustedContrast);
    histogram->show("Histogram Adjusted Contrast", histAdjustedContrast);

    // adjust the image's brightness and display result
    pointOperations->adjustBrightness(imgGray, imgAdjustedBrightness, 50);
    cv::imshow("Adjusted Brightness", imgAdjustedBrightness);
    // calculate and show histogram
    histogram->calcHist(imgAdjustedBrightness, histAdjustedBrightness);
    histogram->show("Histogram Adjusted Brightness", histAdjustedBrightness);

    // invert the image and display result
    pointOperations->invert(imgGray, imgInverted);
    cv::imshow("Inverted", imgInverted);
    // calculate and show histogram
    histogram->calcHist(imgInverted, histInverted);
    histogram->show("Histogram Inverted", histInverted);

    // quantize the image
    pointOperations->quantize(imgGray, imgQuantized, 3);
    cv::imshow("Quantized", imgQuantized);
     // calculate and show histogram
    histogram->calcHist(imgQuantized, histQuantized);
    histogram->show("Histogram Quantized", histQuantized);

    
    // end processing /////////////////////////////////////////////////////////////
    
    //wait for key pressed
    cv::waitKey();
  
    return 0;
}

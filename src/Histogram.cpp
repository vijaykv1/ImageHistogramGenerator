#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Histogram.h"


Histogram::Histogram(){
    histSize = 256;
}

Histogram::~Histogram(){}

////////////////////////////////////////////////////////////////////////////////////
// compute Histogram using the OpenCV function - only for reference
////////////////////////////////////////////////////////////////////////////////////
void Histogram::calcHist_cv(const cv::Mat &input, cv::Mat &hist)
{
    float range[2] = {0.0, (float)(histSize - 1)};   // range of pixel values (min and max value)

    // call OpenCV function for a grayscale image
    cv::calcHist(&input, 1, new int(0), cv::Mat(), hist, 1, &histSize, new const float*(range));
}

///////////////////////////////////////////////////////////////////////////////
// compute Histogram by looping over the elements
///////////////////////////////////////////////////////////////////////////////
void Histogram::calcHist(const cv::Mat &input, cv::Mat &hist)
{

    hist.create(1, histSize, CV_32F);   // one row with 256 values (0 - 255) and we use float values (CV_32F)
    hist.setTo(cv::Scalar(0.0));        // set all values to zero

    // for pointer access of the cv::Mat hist
    float *pHist = hist.ptr<float>(0);

    int rows = input.rows;
    int cols = input.cols;

    if (input.isContinuous()){
        cols = rows*cols;
        rows = 1;
    }

    for (int check_value = 0 ; check_value < 256 ; check_value++)
     {
        int checker_counter;
        uchar check_ticker = check_value; // type cast to uchar value
        for (int r = 0; r < rows ; ++r)
        {
            const uchar *pRow = input.ptr<uchar>(r); // pointer to the rows
            for (int c = 0; c < cols ; ++c)
            {
                if (*pRow == check_ticker){
                    checker_counter++; // INCREMENT ON EACH VALUE'S AVAILABILITY
                }
                ++pRow; // increment to the next element in the row.
            }
        }
        // each element of the row has been covered ... lets record the observations on the pHist pointer
        *pHist = checker_counter; 
        checker_counter = 0; //recorded into the Histogram matrix, lets discard the number and start from the beginning...
        ++pHist; 
     }
}

///////////////////////////////////////////////////////////////////////////////
// calculate statistics of histogram:
//  - hist: input histogram (read the values from this variable)
//  - min: bin with lowest intensity (darkest pixel)
//  - max: bin with highest intensity (brightest pixel)
//  - mean: mean brightness level of the picture
///////////////////////////////////////////////////////////////////////////////
void Histogram::calcStats(const cv::Mat &hist, uchar &min, uchar &max, uchar &mean)
{
    // find minimum
    // use variable "min" for your result
    // min = 0;
    const float *pHistmin = hist.ptr<float>(0); // only one row for the histogram
    for (int checker_value = 0 ; checker_value < 256 ; checker_value++)
    {
        if (*pHistmin > 0)
        {
            min = checker_value; //the first hit non zero pHist value should be the least available value in the picture.
            break; // break the loop here as we got the minimum value that we wanted ! 
        }
        else
        {
            ++pHistmin;
        }

    } //loop for min

    // find maximum
    // use variable "max" for your result
    // max = 255;
    for (int checker_value = 0 ; checker_value < 256 ; checker_value++)
    {
        if(*pHistmin > 0)
        {
            max = checker_value;
            ++pHistmin;
        }
    }    // continue loop until the last value with non zero has been populated ...
    
    // calculate mean
    // use variable "max" for your result
    // mean = 127;
    
    const float *pHistmean = hist.ptr<float>(0);
    for (int checker_value = 0 ; checker_value < 256 ; checker_value++)
    {
        mean = mean + checker_value * (*pHistmean); 
        ++pHistmean;
    }
}


///////////////////////////////////////////////////////////////////////////////
// display Histogram as bar graph
///////////////////////////////////////////////////////////////////////////////
void Histogram::show(const cv::string &winname, const cv::Mat &hist)
{
    // get min and max bin values
    double minVal = 0, maxVal = 0;
    cv::minMaxLoc(hist, &minVal, &maxVal);

    // create new image for displaying the histogram
    cv::Mat histImg(histSize, histSize, CV_8U, cv::Scalar(255));

    // set maximum height of the bars to 90% of the image height
    int maxBinHeight = (int)(0.9 * histSize);

    // draw the bars
    for (int i = 0; i < histSize; ++i){

        float binVal = hist.at<float>(i);
        int binHeight = (int) ( (binVal * maxBinHeight) / maxVal);

        cv::line(histImg, cv::Point(i, histSize),
                          cv::Point(i, histSize - binHeight),
                          cv::Scalar::all(0));

    }

    // show the histogram in a named window
    cv::imshow(winname, histImg);
}

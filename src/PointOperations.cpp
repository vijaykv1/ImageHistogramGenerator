#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#include "PointOperations.h"


PointOperations::PointOperations()
{}

PointOperations::~PointOperations()
{}

////////////////////////////////////////////////////////////////////////////////////
// adjust the contrast of an image by alpha around center
////////////////////////////////////////////////////////////////////////////////////
void PointOperations::adjustContrast(cv::Mat &input, cv::Mat &output, float alpha, uchar center)
{
    int rows = input.rows;
    int cols = input.cols;

    output.release();
    output.create(rows, cols, CV_8U);

    if(input.isContinuous())
    {
        cols = rows*cols; 
        rows = 1;
    }

    for (int r = 0 ; r < rows ; ++r)
    {
        const uchar *pRowInput = input.ptr<uchar>(r);
        uchar *pRowOutput = output.ptr<uchar>(r);
        for (int c = 0 ; c < cols ; ++c)
        {
            *pRowOutput = (alpha * ((*pRowInput) - center)) + center; // formulae for calculating the new contrast image
            ++pRowOutput;
            ++pRowInput; 
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// adjust the brightness of an image by alpha
////////////////////////////////////////////////////////////////////////////////////
void PointOperations::adjustBrightness(cv::Mat &input, cv::Mat &output, int alpha)
{
    int rows = input.rows;
    int cols = input.cols;

    output.release();
    output.create(rows, cols, CV_8U);

    if (input.isContinuous())
    {
        cols = rows * cols;
        rows = 1;
    }
    
    for (int r = 0 ; r< rows ;++r)
    {
        const uchar *pRowInput = input.ptr<uchar>(r); 
        uchar *pRowOutput = output.ptr<uchar>(r);
        for (int c = 0 ; c < cols ; ++c) 
        {
            *pRowOutput = *pRowInput + alpha; //add the alpha to each and every row !
            if (*pRowOutput > 255)
            {
                *pRowOutput = 255; 
            }
            ++pRowInput;
            ++pRowOutput;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// inversion of an image
////////////////////////////////////////////////////////////////////////////////////
void PointOperations::invert(cv::Mat &input, cv::Mat &output)
{
    int rows = input.rows;
    int cols = input.cols;

    output.release();
    output.create(rows, cols, CV_8U);

    if (input.isContinuous()){
        cols = rows*cols;
        rows = 1; 
    }

    for (int r = 0; r< rows ; ++r)
    {
        const uchar *pRowInput = input.ptr<uchar>(r);
        uchar *pRowOutput = output.ptr<uchar>(r);  
        for (int c = 0; c < cols ; ++c)
        {
            *pRowOutput = 255 - (*pRowInput); // flip !! 
            ++pRowInput; 
            ++pRowOutput;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// quantization of an image with n bits
////////////////////////////////////////////////////////////////////////////////////
void PointOperations::quantize(cv::Mat &input, cv::Mat &output, int n)
{
    int rows = input.rows;
    int cols = input.cols;

    output.release();
    output.create(rows, cols, CV_8U);

    if (input.isContinuous())
    {
        cols = cols * rows; 
        rows = 1; 
    }

    // create the scalings 
    // original image bits --> Taking as default => 8 bits i.e. 256 
    int k = 256; 
    int B = std::pow(2,n);
    float interval_size = k/B;  

    for (int r = 0 ; r < rows ; ++r ) 
    {
        const uchar *pRowInput = input.ptr<uchar> (r);
        uchar *pRowOutput = output.ptr<uchar>(r);

        for (int c = 0 ; c < cols ; ++c )
        {
            //float indexCalc = (*pRowInput * (1/interval_size));
            for (int interval_num = 0 ; interval_num < B ; interval_num++)
            {
                if ((*pRowInput >= (interval_num*interval_size)) && (*pRowInput < (interval_num+1) * interval_size))
                {
                    *pRowOutput = interval_size * interval_num;
                }
            }
            
            ++pRowOutput;
            ++pRowInput;
        }
    }
}

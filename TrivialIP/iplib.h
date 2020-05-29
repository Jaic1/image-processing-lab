#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const int hist_width = 512, hist_height = 420, bin_width = hist_width / 256;

// Applies an affine transformation to an image.
//
// no flags, default to INTER_LINEAR, i.e., bilinear interpolation
// M.type() must be CV_64FC1
void warpAffine(const cv::Mat&, cv::Mat&, const cv::Mat&, cv::Size&);

// calculate the histogram of a grey scale image
// return a Mat that can be directly plotted
cv::Mat cal_hist_grey(const cv::Mat&);

// histogram equalization for grey scale image
// only support CV_8U type now, i.e., 0..255
cv::Mat histogram_equalization_grey(const cv::Mat&);

// histogram equalization for color image
// algorithm: rgb->hsi->rgb
// only support CV_8U depth now, i.e., 0..255
cv::Mat histogram_equalization_color_hsi(const cv::Mat&);

// create mosaic image beforehand
void createMosaicImage(cv::Mat, cv::Mat&, int range);

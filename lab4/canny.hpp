#ifndef CANNY_H
#define CANNY_H

#include <math.h>
#include <assert.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>

// Finds edges in an image using the Canny algorithm.
void Canny(const cv::Mat& image, cv::Mat& edges, double threshold1, double threshold2);

#endif

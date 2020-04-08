#ifndef HOUGH_H
#define HOUGH_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <math.h>

// Finds lines in a binary image using the standard Hough transform.
// note: the image must be an 8-bit, single-channel binary source image
void HoughLines(const cv::Mat& image, std::vector<cv::Vec2d>& lines,
	double rho, double theta, int threshold);

#endif

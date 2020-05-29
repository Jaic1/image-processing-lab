#pragma once

#include <QtWidgets>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

// passed-in image might be modified
QImage Mat2QImage(cv::Mat& image);

cv::Mat QImage2Mat(QImage& image);

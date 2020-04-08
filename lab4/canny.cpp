#include "canny.hpp"

static inline uchar cut(double value) {
	int _value = value + 0.5;
	if (_value < 0)
		_value = 0;
	if (_value > 255)
		_value = 255;
	return (uchar)_value;
}

// Blurs an image using a Gaussian filter.
static void GaussianBlur(const cv::Mat& src, cv::Mat& dst, cv::Size ksize, double sigma) {
	int row = src.rows;
	int col = src.cols;
	int channel = src.channels();
	dst = cv::Mat::zeros(row, col, channel == 1 ? CV_8UC1 : CV_8UC3);
	
	// kernel
	int pad_i = ksize.height / 2, pad_j = ksize.width / 2;
	double sigma_square = sigma * sigma;
	double sum = 0;
	double** kernel = new double* [ksize.height];
	for (int i = 0; i < ksize.height; i++) {
		kernel[i] = new double[ksize.width];
	}
	for (int i = 0; i < ksize.height; i++) {
		for (int j = 0; j < ksize.width; j++) {
			int x = j - pad_j;
			int y = i - pad_i;
			kernel[i][j] = 1 / (2 * CV_PI * sigma_square) * exp(-(x * x + y * y) / (2 * sigma_square));
			sum += kernel[i][j];
		}
	}
	for (int i = 0; i < ksize.height; i++) {
		for (int j = 0; j < ksize.width; j++) {
			kernel[i][j] /= sum;
		}
	}

	// filter
	if (channel == 1) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				double value = 0;

				for (int di = -pad_i; di <= pad_i; di++) {
					for (int dj = -pad_j; dj <= pad_j; dj++) {
						if ((i + di >= 0) && (i + di < row) && (j + dj >= 0) && (j + dj < col))
							value += src.at<uchar>(i + di, j + dj) * kernel[pad_i + di][pad_j + dj];
					}
				}

				dst.at<uchar>(i, j) = cut(value);
			}
		}
	}
	else {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				for (int c = 0; c < channel; c++) {
					double value = 0;

					for (int di = -pad_i; di <= pad_i; di++) {
						for (int dj = -pad_j; dj <= pad_j; dj++) {
							if ((i + di >= 0) && (i + di < row) && (j + dj >= 0) && (j + dj < col))
								value += src.at<cv::Vec3b>(i + di, j + dj)[c] * kernel[pad_i + di][pad_j + dj];
						}
					}

					dst.at<uchar>(i, j) = cut(value);
				}
			}
		}
	}
	
	// delete kernel
	for (int i = 0; i < ksize.height; i++) {
		delete[] kernel[i];
	}
	delete[] kernel;
}

// Calculates the first order image derivative in both xand y using a Sobel operator.
static void spatialGradient(const cv::Mat& src, cv::Mat& dx, cv::Mat& dy) {
	const int ksize = 3;

	int row = src.rows;
	int col = src.cols;
	int channel = src.channels();
	int pad = ksize / 2;
	double kernel_x[ksize][ksize] = { {1,0,-1}, {2,0,-2}, {1,0,-1} };
	double kernel_y[ksize][ksize] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

	dx = cv::Mat::zeros(row, col, CV_8UC1);
	dy = cv::Mat::zeros(row, col, CV_8UC1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double value_x = 0;
			double value_y = 0;

			for (int di = -pad; di <= pad; di++) {
				for (int dj = -pad; dj <= pad; dj++) {
					if ((i + di >= 0) && (i + di < row) && (j + dj >= 0) && (j + dj < col)) {
						value_x += src.at<uchar>(i + di, j + dj) * kernel_x[pad + di][pad + dj];
						value_y += src.at<uchar>(i + di, j + dj) * kernel_y[pad + di][pad + dj];
					}
				}
			}

			dx.at<uchar>(i, j) = cut(value_x);
			dy.at<uchar>(i, j) = cut(value_y);
		}
	}
}

// Calculates the gradient and theta given the gradients on x and y axis
static void calGradient(const cv::Mat& dx, const cv::Mat& dy, cv::Mat& grad, cv::Mat& theta) {
	int row = dx.rows;
	int col = dy.cols;

	grad = cv::Mat::zeros(row, col, CV_8UC1);
	theta = cv::Mat::zeros(row, col, CV_8UC1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double _dx = dx.at<uchar>(i, j);
			double _dy = dy.at<uchar>(i, j);
			double angle = atan2(_dy, _dx) * 180 / CV_PI;

			if (angle < -22.5)
				angle += 180;
			if (angle >= 157.5)
				angle -= 180;
			
			if (angle < 22.5) {
				theta.at<uchar>(i, j) = 0;
			}
			else if (angle < 67.5) {
				theta.at<uchar>(i, j) = 45;
			}
			else if (angle < 112.5) {
				theta.at<uchar>(i, j) = 90;
			}
			else {
				theta.at<uchar>(i, j) = 135;
			}

			grad.at<uchar>(i, j) = cut(sqrt(_dx * _dx + _dy * _dy));
		}
	}
}

// Non-maximum suppression as an edge thinning technique.
static void non_maximum_suppression(cv::Mat& grad, const cv::Mat& theta) {
	int row = grad.rows;
	int col = grad.cols;

	cv::Mat grad_out = cv::Mat::zeros(row, col, CV_8UC1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int angle = theta.at<uchar>(i, j);
			int di1, dj1, di2, dj2;

			if (angle == 0) {
				di1 = 0;
				dj1 = -1;
				di2 = 0;
				dj2 = 1;
			}
			else if (angle == 45) {
				di1 = 1;
				dj1 = -1;
				di2 = -1;
				dj2 = 1;
			}
			else if (angle == 90) {
				di1 = -1;
				dj1 = 0;
				di2 = 1;
				dj2 = 0;
			}
			else {
				di1 = -1;
				dj1 = -1;
				di2 = 1;
				dj2 = 1;
			}

			if (i == 0) {
				di1 = std::max(di1, 0);
				di2 = std::max(di2, 0);
			}
			if (i == (row - 1)) {
				di1 = std::min(di1, 0);
				di2 = std::min(di2, 0);
			}
			if (j == 0) {
				dj1 = std::max(dj1, 0);
				dj2 = std::max(dj2, 0);
			}
			if (j == (col - 1)) {
				dj1 = std::min(dj1, 0);
				dj2 = std::min(dj2, 0);
			}
			

			if (grad.at<uchar>(i, j) == std::max(
				std::max(grad.at<uchar>(i, j), grad.at<uchar>(i + di1, j + dj1)),
				grad.at<uchar>(i + di2, j + dj2)))
			{
				grad_out.at<uchar>(i, j) = grad.at<uchar>(i, j);
			}
		}
	}

	grad = grad_out.clone();
}

// Double threshold
static void hysteresis(cv::Mat& grad, int high_threshold, int low_threshold) {
	int row = grad.rows;
	int col = grad.cols;

	cv::Mat grad_out = cv::Mat::zeros(row, col, CV_8UC1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int _grad = grad.at<uchar>(i, j);

			if (_grad >= high_threshold) {
				grad_out.at<uchar>(i, j) = 255;
			}
			else if (_grad >= low_threshold) {
				// blob analysis - look at its 8-connected neighborhood
				for (int di = -1; di <= 1; di++) {
					for (int dj = -1; dj <= 1; dj++) {
						if (i + di < 0 || i + di >= row || j + dj < 0 || j + dj >= col)
							continue;
						if (grad.at<uchar>(i + di, j + dj) >= high_threshold) {
							grad_out.at<uchar>(i, j) = 255;
							break;
						}
					}
				}
			}
		}
	}

	grad = grad_out.clone();
}

void Canny(const cv::Mat& image, cv::Mat& edges, double threshold1, double threshold2) {
	cv::Mat image_ir;
	cv::Mat grad_x, grad_y;
	cv::Mat theta;

	// 1. use gaussian filter to smooth the input image
	GaussianBlur(image, image_ir, cv::Size(5, 5), 1.4);

	// 2. use sobel filter to calculate the gradient
	spatialGradient(image, grad_x, grad_y);
	calGradient(grad_x, grad_y, edges, theta);

	// 3. use non-maximum suppression
	non_maximum_suppression(edges, theta);

	// 4. hysteresis
	hysteresis(edges, threshold2, threshold1);
}

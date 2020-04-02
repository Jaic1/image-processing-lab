#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Applies an affine transformation to an image.
//
// no flags, default to INTER_LINEAR, i.e., bilinear interpolation
// M.type() must be CV_64FC1
void warpAffine(const cv::Mat&, cv::Mat&, const cv::Mat&, cv::Size&);

int main() {
	const char *operation[] = { "translation.jpg", "rotation.jpg", "scale.jpg", "mirror.jpg" };
	std::string img_path;
	std::cout << "please input image: ";
	std::cin >> img_path;

	cv::Mat img_original = cv::imread(img_path);
	cv::namedWindow("original", cv::WINDOW_NORMAL);
	cv::imshow("original", img_original);
	cv::waitKey(0);

	while (true) {
		cv::destroyAllWindows();
		std::string cmd;
		std::cout << ">> ";
		std::cin >> cmd;

		if (!cmd.compare("help")) {
			std::cout << "1: translation" << std::endl;
			std::cout << "2: rotation" << std::endl;
			std::cout << "3: scale" << std::endl;
			std::cout << "4: mirror" << std::endl;
			continue;
		}

		if (!cmd.compare("exit")) {
			cv::destroyAllWindows();
			break;
		}

		int cmd_num = std::stoi(cmd);
		cv::Mat img_dst;
		cv::Size img_dst_size;
		cv::Mat M = cv::Mat::zeros(2, 3, CV_64FC1);
		switch (cmd_num)
		{
		case 1:
			// translation
			int dx, dy;
			std::cout << "x axis: ";
			std::cin >> dx;
			std::cout << "y axis: ";
			std::cin >> dy;
			M.at<double>(0, 0) = 1;
			M.at<double>(0, 2) = dx;
			M.at<double>(1, 1) = 1;
			M.at<double>(1, 2) = dy;
			break;
		case 2:
			// rotation
			double theta, lx, ly;
			double co, si;
			std::cout << "angel: ";
			std::cin >> theta;
			theta = theta * M_PI / 180;
			co = cos(theta);
			si = sin(theta);
			lx = img_original.cols;
			ly = img_original.rows;
			M.at<double>(0, 0) = co;
			M.at<double>(0, 1) = -si;
			M.at<double>(0, 2) = (lx + ly * si - lx * co) / 2;
			M.at<double>(1, 0) = si;
			M.at<double>(1, 1) = co;
			M.at<double>(1, 2) = (ly - ly * co - lx * si) / 2;
			break;
		case 3:
			// scale
			double x_factor, y_factor;
			std::cout << "scale factor for x axis: ";
			std::cin >> x_factor;
			std::cout << "scale factor for y axis: ";
			std::cin >> y_factor;
			M.at<double>(0, 0) = x_factor;
			M.at<double>(1, 1) = y_factor;
			break;
		case 4:
			// mirror
			M.at<double>(0, 0) = -1;
			M.at<double>(0, 2) = img_original.cols;
			M.at<double>(1, 1) = 1;
			break;
		default:
			std::cerr << "unknown command!" << std::endl;
			continue;
		}
		warpAffine(img_original, img_dst, M, img_dst_size);
		cv::imwrite(operation[cmd_num - 1], img_dst);
		cv::namedWindow(operation[cmd_num - 1], cv::WINDOW_NORMAL);
		cv::imshow(operation[cmd_num - 1], img_dst);
		cv::waitKey(0);
	}

	return 0;
}

void warpAffine(const cv::Mat& src, cv::Mat& dst, const cv::Mat& M, cv::Size& dsize) {
	const int eps = 1e-9;
	int row = src.rows;
	int col = src.cols;
	int channel = src.channels();

	if (channel != 3) {
		std::cerr << "src image's num of channels is not 3!" << std::endl;
		return;
	}

	// some coefficients
	double a = M.at<double>(0, 0);
	double b = M.at<double>(0, 1);
	double c = M.at<double>(1, 0);
	double d = M.at<double>(1, 1);
	double tx = M.at<double>(0, 2);
	double ty = M.at<double>(1, 2);
	double det = a * d - b * c;

	// resize dst image
	int resized_row = (int)(row * sqrt(c * c + d * d) + eps);
	int resized_col = (int)(col * sqrt(a * a + b * b) + eps);
	dst = cv::Mat::zeros(resized_row, resized_col, CV_8UC3);

	// parameters for loop
	int _x, _y;
	double dx, dy;
	double x_original, y_original;

	// affine transformation
	// i is on y axis, j is on x axis
	for (int i = 0; i < resized_row; i++) {
		for (int j = 0; j < resized_col; j++) {
			// original point
			// -1 for binear interpolation
			x_original = (d * j - b * i + ty * b - tx * d) / det;
			y_original = (a * i - c * j + tx * c - ty * a) / det;
			if ((x_original < 0) || (x_original >= col-1)) {
				continue;
			}
			if ((y_original < 0) || (y_original >= row-1)) {
				continue;
			}

			// binear interpolation
			_x = (int)x_original;
			_y = (int)y_original;
			dx = x_original - _x;
			dy = y_original - _y;
			for (int c = 0; c < channel; c++) {
				dst.at<cv::Vec3b>(i, j)[c] = (1 - dx) * (1 - dy) * src.at<cv::Vec3b>(_y, _x)[c] +
					dx * (1 - dy) * src.at<cv::Vec3b>(_y, _x + 1)[c] +
					(1 - dx) * dy * src.at<cv::Vec3b>(_y + 1, _x)[c] +
					dx * dy * src.at<cv::Vec3b>(_y + 1, _x + 1)[c];
			}
		}
	}
}
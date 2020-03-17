#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <algorithm>
#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

cv::Mat histogram_equalization_grey(cv::Mat&);
cv::Mat histogram_equalization_color_hsi(cv::Mat&);
cv::Mat cal_hist_grey(cv::Mat&);
cv::Mat cal_hist_color(cv::Mat&);

const int hist_width = 512, hist_height = 420, bin_width = hist_width / 256;

int main() {
	// for grey scale image
	cv::Mat grey[2], grey_out;
	cv::Mat hist[3], hist_out;
	grey[0] = cv::imread("grey.png", cv::IMREAD_GRAYSCALE);
	grey[1] = histogram_equalization_grey(grey[0]);
	cv::hconcat(grey, 2, grey_out);
	cv::imshow("grey", grey_out);
	cv::imwrite("grey_out.png", grey_out);
	hist[0] = cal_hist_grey(grey[0]);
	hist[1] = cv::Mat(hist_height, 10, CV_8UC3, cv::Scalar(144, 238, 144));
	hist[2] = cal_hist_grey(grey[1]);
	cv::hconcat(hist, 3, hist_out);
	cv::imshow("灰度直方图均衡化前后对比（左：均衡化前，右：均衡化后）", hist_out);
	cv::imwrite("hist_out.png", hist_out);

	// for color image
	cv::Mat color[2], color_out;
	color[0] = cv::imread("color.jpg", cv::IMREAD_COLOR);
	color[1] = histogram_equalization_color_hsi(color[0]);
	cv::hconcat(color, 2, color_out);
	cv::imshow("color output", color_out);
	cv::imwrite("color_out.jpg", color_out);

	std::cout << "Press any key to exit.\n";
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

// histogram equalization for grey scale image
// only support CV_8U type now, i.e., 0..255
cv::Mat histogram_equalization_grey(cv::Mat& img) {
	if (img.type() != CV_8U) {
		std::cout << "histogram_equalization_grey: img's type is not CV_8U\n";
		return img;
	}

	const int min = 0, max = 255;
	int value[max + 1] = { 0 };
	double hist[max + 1];
	int row = img.rows, col = img.cols;

	// count the times each grey value occurs
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			value[img.at<uchar>(i, j)]++;
		}
	}

	// calculate P(f) and C(f)
	for (int i = 0; i <= max; i++) {
		hist[i] = (double)value[i] / (row * col);
		if (i > 0)
			hist[i] += hist[i - 1];
	}

	// calculate the mapping g function
	for (int i = 0; i <= max; i++) {
		value[i] = (int)((max - min) * hist[i] + min + 0.5);
	}

	// result
	cv::Mat ret = cv::Mat(row, col, CV_8U);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			ret.at<uchar>(i, j) = value[img.at<uchar>(i, j)];
		}
	}

	return ret;
}

// histogram equalization for color image
// algorithm: rgb->hsi->rgb
// only support CV_8U depth now, i.e., 0..255
cv::Mat rgb_to_hsi(cv::Mat&);
cv::Mat hsi_to_rgb(cv::Mat&);
cv::Mat histogram_equalization_color_hsi(cv::Mat& img) {
	if (img.type() != CV_8UC3) {
		std::cout << "histogram_equalization_color_hsi: img's not CV_8UC3\n";
		return img;
	}

	// rgb to hsi
	cv::Mat hsi = rgb_to_hsi(img);

	// histogram equalization for I channel
	cv::Mat hsi_channels[3], i_chan;
	cv::Mat hist[3], hist_out;
	cv::split(hsi, hsi_channels);
	hsi_channels[2] *= 255;
	hsi_channels[2].convertTo(i_chan, CV_8U);
	hist[0] = cal_hist_grey(i_chan);
	i_chan = histogram_equalization_grey(i_chan);
	hist[2] = cal_hist_grey(i_chan);
	i_chan.convertTo(hsi_channels[2], CV_64F);
	hsi_channels[2] *= 1.0 / 255;
	cv::merge(hsi_channels, 3, hsi);

	// plot histogram before and after transformation
	hist[1] = cv::Mat(hist_height, 10, CV_8UC3, cv::Scalar(144, 238, 144));
	cv::hconcat(hist, 3, hist_out);
	cv::imshow("亮度通道直方图均衡化前后对比（左：均衡化前，右：均衡化后）", hist_out);
	cv::imwrite("hist_i_out.png", hist_out);

	// hsi to rgb
	cv::Mat rgb = hsi_to_rgb(hsi);

	return rgb;
}

// convert rgb to hsi
// rgb is assumed to be ranging from 0 to 255
// return hsi that is normalized
cv::Mat rgb_to_hsi(cv::Mat& rgb) {
	int row = rgb.rows, col = rgb.cols;
	cv::Mat hsi = cv::Mat(row, col, CV_64FC3);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			cv::Vec3d hsi_vec;
			cv::Vec3b rgb_vec = rgb.at<cv::Vec3b>(i, j);
			uchar min = std::min(std::min(rgb_vec[0], rgb_vec[1]), rgb_vec[2]);
			int sum = rgb_vec[0] + rgb_vec[1] + rgb_vec[2];
			int rg = rgb_vec[2] - rgb_vec[1], rb = rgb_vec[2] - rgb_vec[0], gb = rgb_vec[1] - rgb_vec[0];

			// calculate hsi value
			if (rg == 0 && rb == 0) {
				hsi_vec[0] = 0;
				hsi_vec[1] = 0;
			}
			else {
				hsi_vec[0] = acos((rg + rb) / (2 * sqrt(rg * rg + rb * gb)));
				hsi_vec[0] = gb >= 0 ? hsi_vec[0] : 2 * M_PI - hsi_vec[0];
				hsi_vec[1] = 1 - (double)(3 * min) / sum;
			}
			hsi_vec[2] = (double)sum / (3 * 255);

			//std::cout << rg << " " << rb << " " << gb << " " << hsi_vec[0] << std::endl;
			hsi.at<cv::Vec3d>(i, j) = hsi_vec;
		}
	}

	return hsi;
}

// convert hsi to rgb
// hsi is assumed to be normalized
// return rgb that ranges from 0 to 255
cv::Mat hsi_to_rgb(cv::Mat& hsi) {
	int row = hsi.rows, col = hsi.cols;
	cv::Mat rgb = cv::Mat::zeros(row, col, CV_8UC3);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double H = hsi.at<cv::Vec3d>(i, j)[0], S = hsi.at<cv::Vec3d>(i, j)[1], I = hsi.at<cv::Vec3d>(i, j)[2];
			double r, g, b;

			if (H >= 0 && H < 2 * M_PI / 3) {
				b = (1 - S) / 3;
				r = (1 + S * cos(H) / cos(M_PI / 3 - H)) / 3;
				g = 1 - r - b;
			}
			else if (H >= 2 * M_PI / 3 && H < 4 * M_PI / 3) {
				H -= 2 * M_PI / 3;
				r = (1 - S) / 3;
				g = (1 + S * cos(H) / cos(M_PI / 3 - H)) / 3;
				b = 1 - r - g;
			}
			else if (H >= 4 * M_PI / 3 && H <= 2 * M_PI) {
				H -= 4 * M_PI / 3;
				g = (1 - S) / 3;
				b = (1 + S * cos(H) / cos(M_PI / 3 - H)) / 3;
				r = 1 - g - b;
			}
			else {
				std::cout << "out of range, H = " << H << std::endl;
				continue;
			}

			rgb.at<cv::Vec3b>(i, j)[0] = (uchar)(3 * I * b * 255);
			rgb.at<cv::Vec3b>(i, j)[1] = (uchar)(3 * I * g * 255);
			rgb.at<cv::Vec3b>(i, j)[2] = (uchar)(3 * I * r * 255);
		}
	}

	return rgb;
}

// calculate the histogram of a grey scale image
// return a Mat that can be directly plotted
cv::Mat cal_hist_grey(cv::Mat& img) {
	cv::Mat hist_img = cv::Mat(hist_height, hist_width, CV_8UC3, cv::Scalar(225, 228, 255));
	int frequency[256] = {0};
	double hist[256], hist_max;
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			frequency[img.at<uchar>(i, j)]++;
		}
	}

	hist_max = -1;
	for (int i = 0; i <= 255; i++) {
		hist[i] = (double)frequency[i] / (img.rows * img.cols);
		if (hist[i] > hist_max)
			hist_max = hist[i];
	}

	for (int i = 0; i < 255; i++) {
		cv::line(hist_img, cv::Point(i * bin_width, (int)(hist_height * (1 - hist[i] / hist_max))),
			cv::Point((i + 1) * bin_width, (int)(hist_height * (1 - hist[i + 1] / hist_max))),
			cv::Scalar(205, 0, 0), 2, cv::LINE_AA);
	}

	return hist_img;
}
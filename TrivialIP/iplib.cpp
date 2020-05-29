#include "iplib.h"

// Applies an affine transformation to an image.
//
// no flags, default to INTER_LINEAR, i.e., bilinear interpolation
// M.type() must be CV_64FC1
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
			if ((x_original < 0) || (x_original >= col - 1)) {
				continue;
			}
			if ((y_original < 0) || (y_original >= row - 1)) {
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

// calculate the histogram of a grey scale image
// return a Mat that can be directly plotted
cv::Mat cal_hist_grey(const cv::Mat& img) {
	cv::Mat hist_img = cv::Mat(hist_height, hist_width, CV_8UC3, cv::Scalar(225, 228, 255));
	int frequency[256] = { 0 };
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

// histogram equalization for grey scale image
// only support CV_8U type now, i.e., 0..255
cv::Mat histogram_equalization_grey(const cv::Mat& img) {
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

// convert rgb to hsi
// rgb is assumed to be ranging from 0 to 255
// return hsi that is normalized
static cv::Mat rgb_to_hsi(const cv::Mat& rgb) {
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
static cv::Mat hsi_to_rgb(const cv::Mat& hsi) {
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


// histogram equalization for color image
// algorithm: rgb->hsi->rgb
// only support CV_8U depth now, i.e., 0..255
cv::Mat histogram_equalization_color_hsi(const cv::Mat& img) {
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
	cv::imshow("hist out", hist_out);

	// hsi to rgb
	cv::Mat rgb = hsi_to_rgb(hsi);

	return rgb;
}

// create mosaic image beforehand
void createMosaicImage(cv::Mat in, cv::Mat& out, int range) {
	cv::RNG rng;
	int height = in.rows;
	int width = in.cols;
	cv::Mat padding;
	cv::Mat tmp;

	cv::copyMakeBorder(in, padding, 0, range - in.rows % range, 0, range - in.cols % range, cv::BORDER_REPLICATE);
	tmp = padding.clone();

	for (int row = 0; row < padding.rows; row += range) {
		for (int col = 0; col < padding.cols; col += range) {
			int rand_x = rng.uniform(0, range);
			int rand_y = rng.uniform(0, range);
			cv::Rect rect = cv::Rect(col, row, range, range);
			cv::Mat roi = tmp(rect);
			cv::Scalar color = cv::Scalar(padding.at<cv::Vec3b>(row + rand_y, col + rand_x)[0], \
				padding.at<cv::Vec3b>(row + rand_y, col + rand_x)[1], \
				padding.at<cv::Vec3b>(row + rand_y, col + rand_x)[2]);
			roi.setTo(color);
		}
	}

	out = tmp(cv::Rect(0, 0, width, height)).clone();
}

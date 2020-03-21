#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void medianBlur(cv::Mat& src, cv::Mat& dst, int ksize);

int main() {
	cv::Mat noise = cv::imread("img/noise.jpg", cv::IMREAD_GRAYSCALE);

	// 3 * 3 median filtering
	cv::Mat noise_out_opencv_3x3, noise_out_mine_3x3;
	cv::medianBlur(noise, noise_out_opencv_3x3, 3);
	medianBlur(noise, noise_out_mine_3x3, 3);
	cv::namedWindow("3*3中值滤波(opencv)", cv::WINDOW_NORMAL);
	cv::namedWindow("3*3中值滤波(mine)", cv::WINDOW_NORMAL);
	cv::imshow("3*3中值滤波(opencv)", noise_out_opencv_3x3);
	cv::imshow("3*3中值滤波(mine)", noise_out_mine_3x3);
	cv::imwrite("img/noise_out_opencv_3x3.jpg", noise_out_opencv_3x3);
	cv::imwrite("img/noise_out_mine_3x3.jpg", noise_out_mine_3x3);

	// 5*5, 7*7, 15*15 median filter
	int ksizes[] = {5, 7, 15};
	for (auto ksize : ksizes) {
		cv::Mat noise_out_opencv, noise_out_mine;
		cv::medianBlur(noise, noise_out_opencv, ksize);
		medianBlur(noise, noise_out_mine, ksize);
		std::ostringstream name_opencv, name_mine;
		name_opencv << "img/noise_out_opencv_" << ksize << "x" << ksize << ".jpg";
		name_mine << "img/noise_out_mine_" << ksize << "x" << ksize << ".jpg";
		cv::imwrite(name_opencv.str(), noise_out_opencv);
		cv::imwrite(name_mine.str(), noise_out_mine);
	}

	std::cout << "Press any key to exit.\n";
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

// median filter for grey scale image
// only accept ksize that is odd
void medianBlur(cv::Mat& src, cv::Mat& dst, int ksize) {
	if (src.channels() != 1)
		return;

	int row = src.rows;
	int col = src.cols;
	int pad = (ksize - 1) / 2;
	int* value = new int[ksize * ksize];
	dst = cv::Mat::zeros(row, col, src.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int cnt = 0;

			for (int di = -pad; di <= pad; di++) {
				for (int dj = -pad; dj <= pad; dj++) {
					// use border replicate strategy
					int x = i + di, y = j + dj;
					if (x < 0)
						x = 0;
					if (x >= row)
						x = row - 1;
					if (y < 0)
						y = 0;
					if (y >= col)
						y = col - 1;
					value[cnt++] = (int)(src.at<uchar>(x, y));
				}
			}
			std::sort(value, value + cnt);
			dst.at<uchar>(i, j) = (uchar)value[(cnt - 1) / 2];
		}
	}

	delete[] value;
}
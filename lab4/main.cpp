#include "canny.hpp"
#include "hough.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

cv::Mat DrawLines(const cv::Mat& src, std::vector<cv::Vec2f> lines);

int main() {
	cv::Mat img = cv::imread("hf.jpg", cv::IMREAD_COLOR);
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	// opencv standard api
	cv::Mat img_blured;
	cv::GaussianBlur(img_gray, img_blured, cv::Size(5, 5), 0);

	cv::Mat edges;
	cv::Canny(img_blured, edges, 40, 100);
	cv::namedWindow("edges", cv::WINDOW_NORMAL);
	cv::imshow("edges", edges);
	cv::imwrite("edges_opencv.jpg", edges);

	std::vector<cv::Vec2f> lines;
	cv::HoughLines(edges, lines, 1, CV_PI / 180, 200);
	cv::Mat img_out = DrawLines(img, lines);
	cv::namedWindow("lines", cv::WINDOW_NORMAL);
	cv::imshow("lines", img_out);
	cv::imwrite("hf_lines_opencv.jpg", img_out);

	// my implementation
	// my Canny already contains GaussianBlur
	cv::Mat edges_mine;
	Canny(img_gray, edges_mine, 40, 100);
	cv::namedWindow("edges_mine", cv::WINDOW_NORMAL);
	cv::imshow("edges_mine", edges_mine);
	cv::imwrite("edges_mine.jpg", edges_mine);

	std::vector<cv::Vec2f> lines_mine;
	HoughLines(edges_mine, lines_mine, 1, CV_PI / 180, 140);
	cv::Mat img_out_mine = DrawLines(img, lines_mine);
	cv::namedWindow("lines_mine", cv::WINDOW_NORMAL);
	cv::imshow("lines_mine", img_out_mine);
	cv::imwrite("hf_lines_mine.jpg", img_out_mine);

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}

cv::Mat DrawLines(const cv::Mat& src, std::vector<cv::Vec2f> lines) {
	int row = src.rows;
	int col = src.cols;
	cv::Mat dst = src.clone();

	for (auto line : lines) {
		double rho = line[0], theta = line[1];
		double si = sin(theta), co = cos(theta);

		if (co != 0) {
			for (int i = 0; i < row; i++) {
				int j = (rho - i * si) / co;
				if (j >= 0 && j < col)
					dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
			}
		}

		if (si != 0) {
			for (int j = 0; j < col; j++) {
				int i = (rho - j * co) / si;
				if (i >= 0 && i < row)
					dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
			}
		}
	}

	return dst;
}

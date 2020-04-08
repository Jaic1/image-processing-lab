#include "hough.hpp"

const double eps = 1e-9;
static double RHO_MAX;
static double THETA_MAX;
static int rho_cnt;
static int theta_cnt;

static void HoughVote(const cv::Mat& image, int** votes,
	double thetas[], double rhos[])
{
	int row = image.rows;
	int col = image.cols;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// not edge
			if (image.at<uchar>(i, j) != 255)
				continue;

			for (int tc = 0; tc < theta_cnt; tc++) {
				double theta = thetas[tc] * CV_PI / 180;
				double rho = j * cos(theta) + i * sin(theta);
				int rc = std::lower_bound(rhos, rhos + rho_cnt + 1, rho + eps) - rhos;
				rc--;
				assert(rc >= 0 && rc <= rho_cnt);

				// rho == RHO_MAX, count it into [, RHOMAX) which is the last interval
				if (rho_cnt)
					rc--;
				votes[tc][rc]++;
			}
		}
	}
}

static void HoughInverse(std::vector<cv::Vec2f>& lines, int** votes,
	double thetas[], double rhos[], int threshold)
{
	for (int tc = 0; tc < theta_cnt; tc++) {
		for (int rc = 0; rc < rho_cnt; rc++) {
			if (votes[tc][rc] < threshold)
				continue;

			lines.push_back(cv::Vec2f(rhos[rc], thetas[tc]));
		}
	}
}

// Hough line transformation, the input image is expected to be edge map
// note for implementation: rho could be minus
void HoughLines(const cv::Mat& image, std::vector<cv::Vec2f>& lines,
	double rho, double theta, int threshold)
{
	RHO_MAX = sqrt(image.rows * image.rows + image.cols * image.cols);
	THETA_MAX = 180;
	theta *= 180 / CV_PI;	// tmp fix
	rho_cnt = ((int)(RHO_MAX / rho - eps) + 1) * 2;
	theta_cnt = (int)(THETA_MAX / theta - eps) + 1;

	// create table for theta, rho
	double* thetas = new double[theta_cnt+1];
	double* rhos = new double[rho_cnt+1];
	int** votes = new int* [theta_cnt];
	for (int i = 0; i < theta_cnt; i++) {
		thetas[i] = (i == 0 ? 0 : thetas[i - 1] + theta);
		votes[i] = new int[rho_cnt];
		for (int j = 0; j < rho_cnt; j++)
			votes[i][j] = 0;
	}
	thetas[theta_cnt] = THETA_MAX;
	for (int i = 0; i < rho_cnt/2; i++) {
		rhos[rho_cnt / 2 + i] = (i == 0 ? 0 : rhos[rho_cnt / 2 + i - 1] + rho);
		rhos[rho_cnt / 2 - i] = (i == 0 ? 0 : rhos[rho_cnt / 2 - i + 1] - rho);
	}
	rhos[0] = -RHO_MAX;
	rhos[rho_cnt] = RHO_MAX;

	// 1. iterate through the edge image to vote
	HoughVote(image, votes, thetas, rhos);

	// 2. inverse transformation
	HoughInverse(lines, votes, thetas, rhos, threshold);

	// delete table
	delete[] thetas;
	delete[] rhos;
	for (int i = 0; i < theta_cnt; i++) {
		delete[] votes[i];
	}
	delete[] votes;
}

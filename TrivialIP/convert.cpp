#include "convert.h"

// passed-in image might be modified
QImage Mat2QImage(cv::Mat& image)
{
    QImage qtImg;
    if (!image.empty() && image.depth() == CV_8U) {
        if (image.channels() == 1) {
            qtImg = QImage((const unsigned char*)(image.data),
                image.cols,
                image.rows,
                QImage::Format_Indexed8);
        }
        else {
            cvtColor(image, image, CV_BGR2RGB);
            qtImg = QImage((const unsigned char*)(image.data),
                image.cols,
                image.rows,
                QImage::Format_RGB888);
        }
    }
    return qtImg;
}

cv::Mat QImage2Mat(QImage& image) {

    cv::Mat cvImage;
    switch (image.format()) {
    case QImage::Format_RGB888: {
        cvImage = cv::Mat(image.height(),
            image.width(),
            CV_8UC3,
            image.bits(),
            image.bytesPerLine());
        cv::cvtColor(cvImage, cvImage, CV_RGB2BGR);
        return cvImage;
    }
    case QImage::Format_Indexed8: {
        cvImage = cv::Mat(image.height(),
            image.width(),
            CV_8U,
            image.bits(),
            image.bytesPerLine());
        return cvImage;
    }
    default:
        break;
    }
    return cvImage;
}
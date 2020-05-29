#include "trivialip.h"
#include "stdafx.h"
#include "convert.h"
#include "iplib.h"
#include <iostream>

TrivialIP::TrivialIP(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    cv_cache.clear();
    img_cache.clear();
    img_cache_i = -1;
}

void TrivialIP::on_actionOpen_triggered() {
    QString filename = QFileDialog::getOpenFileName(this,
        tr("select image"), "./", tr("*.png *.jpg;;" "*.txt"));

    cv_cache.clear();
    img_cache.clear();
    cv_cache.push_back(cv::Mat());
    img_cache.push_back(QImage());
    img_cache_i = 0;

    cv::Mat cv_img = cv::imread(filename.toStdString());
    cv_img.copyTo(cv_cache[0]);
    QImage img = Mat2QImage(cv_img);
    img_cache[0] = img.copy();
    ui.image->setPixmap(QPixmap::fromImage(img_cache[0]));
    ui.width_label->setText(QString::number(img.width()));
    ui.height_label->setText(QString::number(img.height()));
}

void TrivialIP::mousePressEvent(QMouseEvent* e) {
    QPoint event_p = e->pos();
    QPoint center_p = ui.centralWidget->pos();
    QPoint area_p = ui.scrollArea->pos();
    QPoint image_p = ui.image->pos();
    QPoint total_p = center_p + area_p + image_p;
    QPoint scroll_p = ui.scrollAreaWidgetContents->pos();
    QPoint p = event_p - total_p - scroll_p;

    if (event_p.x() < total_p.x() || event_p.y() < total_p.y()
        || event_p.x() > ui.scrollArea->width() + area_p.x()
        || event_p.y() > ui.scrollArea->height() + area_p.y()) {
        return;
    }

    if (img_cache_i < 0 || img_cache.empty()) {
        return;
    }

    QRgb color = img_cache[img_cache_i].pixel(p);
    ui.red_label->setText(QString::number(qRed(color)));
    ui.green_label->setText(QString::number(qGreen(color)));
    ui.blue_label->setText(QString::number(qBlue(color)));
}

void TrivialIP::mouseMoveEvent(QMouseEvent* e) {
    if (img_cache_i < 0 || !ui.checkBoxMosaic->isChecked()) {
        return;
    }

    QPoint event_p = e->pos();
    QPoint center_p = ui.centralWidget->pos();
    QPoint area_p = ui.scrollArea->pos();
    QPoint image_p = ui.image->pos();
    QPoint total_p = center_p + area_p + image_p;
    QPoint scroll_p = ui.scrollAreaWidgetContents->pos();
    QPoint p = event_p - total_p - scroll_p;

    if (event_p.x() < total_p.x() || event_p.y() < total_p.y()
        || event_p.x() > ui.scrollArea->width() + area_p.x()
        || event_p.y() > ui.scrollArea->height() + area_p.y()) {
        return;
    }

    int xl, xr, yt, yb;
    cv::Mat& cv_img = cv_cache[img_cache_i];
    xl = (p.x() - mosaic_range <= 0) ? 0 : p.x() - mosaic_range;
    xr = (p.x() + mosaic_range > cv_img.cols) ? cv_img.cols : p.x() + mosaic_range;
    yt = (p.y() - mosaic_range <= 0) ? 0 : p.y() - mosaic_range;
    yb = (p.y() + mosaic_range > cv_img.rows) ? cv_img.rows : p.y() + mosaic_range;

    cv::Rect rect_mask = cv::Rect(xl, yt, xr - xl, yb - yt);
    cv::Mat mosaic_mask = img_mosaic(rect_mask);
    cv::Mat src_mask = cv_img(rect_mask);
    mosaic_mask.copyTo(src_mask);

    // debug
    //cv::imshow("all mosaic", img_mosaic);
    //cv::imshow("tmp mask", cv_img);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
    // debug

    cv::Mat cv_img_tmp = cv_img.clone();
    QImage qimg = Mat2QImage(cv_img_tmp);
    img_cache[img_cache_i] = qimg.copy();
    ui.image->setPixmap(QPixmap::fromImage(img_cache[img_cache_i]));
    ui.width_label->setText(QString::number(img_cache[img_cache_i].width()));
    ui.height_label->setText(QString::number(img_cache[img_cache_i].height()));
}

void TrivialIP::on_pushButtonBack_clicked() {
    if (img_cache_i <= 0) {
        return;
    }

    const QImage& img = img_cache[--img_cache_i];
    ui.image->setPixmap(QPixmap::fromImage(img));
    ui.width_label->setText(QString::number(img.width()));
    ui.height_label->setText(QString::number(img.height()));
}

void TrivialIP::on_pushButtonSave_clicked() {
    if (img_cache_i < 0) {
        return;
    }

    QString filename = QFileDialog::getSaveFileName(this,
        tr("save image"), "./", tr("*.png *.jpg;;" "*.txt"));
    cv::imwrite(filename.toStdString(), cv_cache[img_cache_i]);
}

void TrivialIP::on_pushButtonForward_clicked() {
    if (img_cache_i + 1 >= img_cache.size()) {
        return;
    }

    const QImage& img = img_cache[++img_cache_i];
    ui.image->setPixmap(QPixmap::fromImage(img));
    ui.width_label->setText(QString::number(img.width()));
    ui.height_label->setText(QString::number(img.height()));
}

void TrivialIP::on_pushButtonScale_clicked() {
    cv::Mat img_original = cv_cache[img_cache_i].clone();
    cv::Mat img_dst;
    cv::Size img_dst_size;

    cv::Mat M = cv::Mat::zeros(2, 3, CV_64FC1);
    double x_factor = ui.lineEditScaleX->text().toDouble();
    double y_factor = ui.lineEditScaleY->text().toDouble();
    M.at<double>(0, 0) = x_factor;
    M.at<double>(1, 1) = y_factor;

    warpAffine(img_original, img_dst, M, img_dst_size);

    if (img_cache_i == cv_cache.size() - 1) {
        cv_cache.push_back(cv::Mat());
        img_cache.push_back(QImage());
    }
    
    img_cache_i++;
    img_dst.copyTo(cv_cache[img_cache_i]);
    img_cache[img_cache_i] = Mat2QImage(img_dst).copy();
    ui.image->setPixmap(QPixmap::fromImage(img_cache[img_cache_i]));
    ui.width_label->setText(QString::number(img_cache[img_cache_i].width()));
    ui.height_label->setText(QString::number(img_cache[img_cache_i].height()));
}

void TrivialIP::on_pushButtonRotation_clicked() {
    cv::Mat img_original = cv_cache[img_cache_i].clone();
    cv::Mat img_dst;
    cv::Size img_dst_size;

    cv::Mat M = cv::Mat::zeros(2, 3, CV_64FC1);
    double theta, lx, ly;
    double co, si;
    theta = ui.lineEditRotation->text().toDouble();
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

    warpAffine(img_original, img_dst, M, img_dst_size);

    if (img_cache_i == cv_cache.size() - 1) {
        cv_cache.push_back(cv::Mat());
        img_cache.push_back(QImage());
    }

    img_cache_i++;
    img_dst.copyTo(cv_cache[img_cache_i]);
    img_cache[img_cache_i] = Mat2QImage(img_dst).copy();
    ui.image->setPixmap(QPixmap::fromImage(img_cache[img_cache_i]));
    ui.width_label->setText(QString::number(img_cache[img_cache_i].width()));
    ui.height_label->setText(QString::number(img_cache[img_cache_i].height()));
}

void TrivialIP::on_pushButtonMirror_clicked() {
    cv::Mat img_original = cv_cache[img_cache_i].clone();
    cv::Mat img_dst;
    cv::Size img_dst_size;

    cv::Mat M = cv::Mat::zeros(2, 3, CV_64FC1);
    M.at<double>(0, 0) = -1;
    M.at<double>(0, 2) = img_original.cols;
    M.at<double>(1, 1) = 1;

    warpAffine(img_original, img_dst, M, img_dst_size);

    if (img_cache_i == cv_cache.size() - 1) {
        cv_cache.push_back(cv::Mat());
        img_cache.push_back(QImage());
    }

    img_cache_i++;
    img_dst.copyTo(cv_cache[img_cache_i]);
    img_cache[img_cache_i] = Mat2QImage(img_dst).copy();
    ui.image->setPixmap(QPixmap::fromImage(img_cache[img_cache_i]));
    ui.width_label->setText(QString::number(img_cache[img_cache_i].width()));
    ui.height_label->setText(QString::number(img_cache[img_cache_i].height()));
}

void TrivialIP::on_pushButtonHist_clicked() {
    if (img_cache_i < 0) {
        return;
    }

    cv::Mat img_dst;

    if (cv_cache[img_cache_i].channels() == 1) {
        // for grey scale image
        cv::Mat grey[2];
        cv::Mat hist[3], hist_out;
        grey[0] = cv_cache[img_cache_i];
        grey[1] = histogram_equalization_grey(grey[0]);
        hist[0] = cal_hist_grey(grey[0]);
        hist[1] = cv::Mat(hist_height, 10, CV_8UC3, cv::Scalar(144, 238, 144));
        hist[2] = cal_hist_grey(grey[1]);
        cv::hconcat(hist, 3, hist_out);
        cv::imshow("hist(press any key to exit!)", hist_out);
        grey[1].copyTo(img_dst);
    }
    else {
        // for color image
        cv::Mat color[2], color_out;
        color[0] = cv_cache[img_cache_i];
        color[1] = histogram_equalization_color_hsi(color[0]);
        color[1].copyTo(img_dst);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    if (img_cache_i == cv_cache.size() - 1) {
        cv_cache.push_back(cv::Mat());
        img_cache.push_back(QImage());
    }

    img_cache_i++;
    img_dst.copyTo(cv_cache[img_cache_i]);
    img_cache[img_cache_i] = Mat2QImage(img_dst).copy();
    ui.image->setPixmap(QPixmap::fromImage(img_cache[img_cache_i]));
    ui.width_label->setText(QString::number(img_cache[img_cache_i].width()));
    ui.height_label->setText(QString::number(img_cache[img_cache_i].height()));
}

void TrivialIP::on_checkBoxMosaic_clicked(bool checked) {
    if (img_cache_i < 0) {
        return;
    }

    //this->setMouseTracking(checked);
    //ui.centralWidget->setMouseTracking(checked);
    //ui.scrollArea->setMouseTracking(checked);
    //ui.scrollAreaWidgetContents->setMouseTracking(checked);
    //ui.image->setMouseTracking(checked);
    if (!checked) {
        return;
    }

    if (img_cache_i == cv_cache.size() - 1) {
        cv_cache.push_back(cv::Mat());
        img_cache.push_back(QImage());
    }

    img_cache_i++;
    cv_cache[img_cache_i - 1].copyTo(cv_cache[img_cache_i]);
    img_cache[img_cache_i] = Mat2QImage(cv_cache[img_cache_i - 1].clone()).copy();
    ui.image->setPixmap(QPixmap::fromImage(img_cache[img_cache_i]));
    ui.width_label->setText(QString::number(img_cache[img_cache_i].width()));
    ui.height_label->setText(QString::number(img_cache[img_cache_i].height()));

    createMosaicImage(cv_cache[img_cache_i].clone(), img_mosaic, mosaic_range);
}

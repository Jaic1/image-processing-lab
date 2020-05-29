#pragma once

#include "ui_trivialip.h"
#include <vector>
#include <QtWidgets/QMainWindow>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class TrivialIP : public QMainWindow
{
    Q_OBJECT

public:
    TrivialIP(QWidget *parent = Q_NULLPTR);

protected:
    void mousePressEvent(QMouseEvent* e);
    void mouseMoveEvent(QMouseEvent* e);

private:
    Ui::TrivialIPClass ui;
    std::vector<cv::Mat> cv_cache;
    std::vector<QImage> img_cache;
    int img_cache_i;
    const int mosaic_range = 15;
    cv::Mat img_mosaic;
    cv::RNG rng;

private slots:
    void on_actionOpen_triggered();
    void on_pushButtonBack_clicked();
    void on_pushButtonSave_clicked();
    void on_pushButtonForward_clicked();
    void on_pushButtonScale_clicked();
    void on_pushButtonRotation_clicked();
    void on_pushButtonMirror_clicked();
    void on_pushButtonHist_clicked();
    void on_checkBoxMosaic_clicked(bool);
};

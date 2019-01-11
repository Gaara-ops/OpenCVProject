#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "myhead.h"

#include "myopencvfunc.h"
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

	void ShowImageMat(cv::Mat img,int posindex=0);
	void mousePressEvent(QMouseEvent *e);
private slots:
	//写入数据到yml文件
	void on_actionWrite_triggered();
	//从yml文件读取数据
	void on_actionRead_triggered();
	//显示一张图片
	void on_actionShowImage_triggered();
	//显示一张图片，并进行Blur、Gray、Sobel处理
	void on_actionShowImageDeal_triggered();
	//显示一张图片添加鼠标事件
	void on_actionImageWithMouseEvent_triggered();
	//对图片进行Show histogram、Equalize histogram、Lomography、Cartonize处理
	void on_actionImageUsualHandle_triggered();
	//打开视频
	void on_actionOpenVideo_triggered();
	//通过QImage显示图片
	void on_actionShowOnQImage_triggered();
	//获取Mat每一点像素值
	void on_actionGetPixData_triggered();
	//获取图像每一点颜色到指定R G B指定距离范围内的颜色
	void on_actionGetPixDistance_triggered();
	//使图像更加锐利
	void on_actionShapeImage_triggered();
	//翻转图像
	void on_actionFlipImage_triggered();
	//在一张图片中插入另一张图片
	void on_actionInsertImage_triggered();
	//改变图像明亮度
	void on_horizontalSlider_bright_valueChanged(int value);
	//改变图像对比度
	void on_horizontalSlider_contrast_valueChanged(int value);
	//转变颜色类型
	void on_actionConvertColor_triggered();
	//旋转图像
	void on_RotateBtn_clicked();
	//log transform
	void on_LogTFBtn_clicked();
	//gamma transform
	void on_GammaTFBtn_clicked();
	//金字塔处理(放大,缩小)
	void on_PyramidBtn_clicked();
	//阈值筛选
	void on_ThresholdBtn_clicked();
	//区域增长
	void on_SeedGrowthBtn_clicked();
	//特征提取
	void on_LBPBtn_clicked();

private:
	void TransFormImage(int type);//0:log  1:gamma
	void TestFunc();
	//1.Direct 2.Pointer 3.Iterator 获取图像buffer
	void GetImageData(cv::Mat &image, int n,int type=0);
	//获取图像每一点颜色到指定R G B指定距离范围内的颜色
	void GetPixDistance(Mat &image);
	int GetDistance(const cv::Vec3b &color, const cv::Vec3b& target);
	//使图像更加锐利
	void ShapeImage(const cv::Mat &image, cv::Mat &result);
	//改变图像的对比度和明亮度
	void ChangeBrightAndContrast(double bright,double contrast);
	//均衡图像直方图
	void EqualizesHistogram();
private:
	Ui::MainWindow *ui;
	String m_FileName;
	MyOpencvFunc  m_myFunc;
	Mat m_image;
	cv::Point2i seed_point_;
};

#endif // MAINWINDOW_H

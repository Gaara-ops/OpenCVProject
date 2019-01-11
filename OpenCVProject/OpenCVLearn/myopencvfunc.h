#ifndef MYOPENCVFUNC_H
#define MYOPENCVFUNC_H
#include "myhead.h"
using namespace cv;
using namespace std;
class MyOpencvFunc
{
public:
	MyOpencvFunc();
	//显示一张图片,方式2没有平滑处理,但有其他操作
	void ShowImage(QString filepath,int way=1,bool showSmooth=0);
	//在显示图片是添加鼠标事件
	void ShowImageWithMouse(QString filepath);
	//对图像进行常用处理
	void ImageUsualHandle(QString filepath);
	//打开一段视频
	int OpenVideo(QString filepath);

	//区域增长
	cv::Mat regionGrowFast(const cv::Mat &src,
						   const cv::Point2i seed, int throld);
	//LBP处理(FeatureExtraction)特征提取
	void baseLBP(const cv::Mat src, cv::Mat &dst);
	//cv图像转Qimage
	QImage cvMatToQImage(const cv::Mat& mat);
	//Qimage转cv图像
	cv::Mat QImageTocvMat(QImage image);
	//颜色转换RGB->HSI
	void BGRToHSI(const cv::Mat src, cv::Mat &dst);
};

#endif // MYOPENCVFUNC_H

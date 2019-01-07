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
};

#endif // MYOPENCVFUNC_H

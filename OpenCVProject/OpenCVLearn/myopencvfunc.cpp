#include "myopencvfunc.h"

Mat m_img;
String windowname = "winName";
void grayCallback(int state,void* userData);
void blurCallback(int state,void* userData);
void sobelCallback(int state,void* userData);
//鼠标回调函数
static void onMouse(int event,int x,int y,int ,void* userInput);
//滑动条回调函数
int blurAmount = 15;//滑动条位置
static void onChange(int pos,void* userInput);

void showHistoCallback(int state,void* userData);
void equalizeCallback(int state,void* userData);
void lomoCallback(int state,void* userData);
void cartoonCallback(int state,void* userData);

MyOpencvFunc::MyOpencvFunc()
{
}

void MyOpencvFunc::ShowImage(QString filepath, int way, bool showSmooth)
{
	if(way == 1){
		//将图像文件加载至内存
		IplImage* img = cvLoadImage(filepath.toStdString().c_str());
		//在屏幕上创建一个窗口
		//CV_WINDOW_AUTOSIZE-->根据图像的实际大小自动进行拉伸或缩放
		cvNamedWindow( windowname.c_str(), CV_WINDOW_AUTOSIZE );
		//显示该图像
		cvShowImage( windowname.c_str(), img );
		if(showSmooth){
			///对图像进行平滑处理
			cvNamedWindow( "smooth-out" );
			//当前图像结构的大小,各通道每个像素点的数据类型,通道的总数
			IplImage* out = cvCreateImage(
				cvGetSize(img),
				IPL_DEPTH_8U,
				3
			);
			cvSmooth( img, out, CV_GAUSSIAN, 3, 3 );
			cvShowImage( "smooth-out", out );
		}
		//使程序暂停，等待用户触发一个按键操作
		//设置该函数参数为0或负数时，程序将一直等待用户触发按键操作
		//设为一个正数，则程序将暂停一段时间，时间长为该整数值个毫秒单位
		waitKey(0);
		//释放为该图像文件所分配的内存
		cvReleaseImage( &img );
		//销毁显示图像文件的窗口
		destroyWindow(windowname);
		if(showSmooth){
			cvReleaseImage( &img );
			cvDestroyWindow( "smooth-out" );
		}
	}else if(way == 2){
		//读取原图像,并以矩阵格式存储图像
		String filename = filepath.toStdString();
		m_img = imread(filename);
		namedWindow(windowname);
		createButton("Blur",blurCallback,NULL,QT_CHECKBOX,0);
		createButton("Gray",grayCallback,NULL,QT_RADIOBOX,0);
		createButton("Sobel",sobelCallback,NULL,QT_PUSH_BUTTON,0);
		imshow(windowname,m_img);
		waitKey(0);
		destroyWindow(windowname);
	}
}

void MyOpencvFunc::ShowImageWithMouse(QString filepath)
{
	String filename = filepath.toStdString();
	m_img = imread(filename);
	namedWindow(windowname);

	//创建一个滑动条
	createTrackbar("trackbar",windowname,&blurAmount,30,onChange,&m_img);
	setMouseCallback(windowname,onMouse,&m_img);
	//调用onChange初始化
	onChange(blurAmount,&m_img);

	imshow(windowname,m_img);
	waitKey(0);
	destroyWindow(windowname);
}

void MyOpencvFunc::ImageUsualHandle(QString filepath)
{
	String filename = filepath.toStdString();
	m_img = imread(filename);
	namedWindow(windowname);

	createButton("Show histogram",showHistoCallback,NULL,QT_PUSH_BUTTON,0);
	createButton("Equalize histogram",equalizeCallback,NULL,QT_PUSH_BUTTON,0);
	createButton("Lomography effect",lomoCallback,NULL,QT_PUSH_BUTTON,0);
	createButton("Cartonize effect",cartoonCallback,NULL,QT_PUSH_BUTTON,0);

	imshow(windowname,m_img);
	waitKey(0);
	destroyWindow(windowname);
}

int MyOpencvFunc::OpenVideo(QString filepath)
{
	VideoCapture cap;
	String filename = filepath.toStdString();
	cap.open(filename);
	//检查是否打开成功
	if(!cap.isOpened()){
		return -1;
	}
	namedWindow("video",1);
	for(;;){
		Mat frame;
		//获取视频的每一帧
		cap>>frame;
		imshow("video",frame);
		if(waitKey(60) >=0){
			break;
		}
	}
	//释放资源
	cap.release();
	return 1;
}
//滑动条回调函数
static void onChange(int pos,void* userInput){
	if(pos<=0){
		return;
	}
	Mat imgBlur;
	Mat* img = (Mat*)userInput;
	//应用模糊滤镜
	blur(*img,imgBlur,Size(pos,pos));
	//显示输出
	imshow(windowname,imgBlur);
}
//鼠标回调函数
static void onMouse(int event,int x,int y,int ,void* userInput){
	if(event != EVENT_LBUTTONDOWN)
		return;

	Mat* img = (Mat*)userInput;
	//绘制圆
	circle(*img,Point(x,y),10,Scalar(0,255,0),3);
	//调用模糊处理方法
	onChange(blurAmount,img);
}

void grayCallback(int state, void *userData)
{

	Mat result;
	m_img.copyTo(result);
	cvtColor(result,result,COLOR_BGR2GRAY);
	namedWindow("gray");
	imshow("gray",result);
	waitKey(0);
	destroyWindow("gray");
}

void blurCallback(int state, void *userData)
{
	Mat result;
	m_img.copyTo(result);
	blur(result,result,Size(5,5));
	namedWindow("blur");
	imshow("blur",result);
	waitKey(0);
	destroyWindow("blur");
}

void sobelCallback(int state, void *userData)
{
	Mat result;
	m_img.copyTo(result);
	Sobel(result,result,CV_8U,1,1);
	namedWindow("sobel");
	imshow("sobel",result);
	waitKey(0);
	destroyWindow("sobel");
}

void showHistoCallback(int state,void* userData){
	//将图像分为3个通道BGR
	vector<Mat> bgr;
	split(m_img,bgr);
	//创建256个子区间的直方图
	int numbins = 256;
	//设置bgr范围(0--255)
	float range[] = {0,256};
	const float* histRange = {range};

	Mat b_hist,g_hist,r_hist;
	calcHist(&bgr[0],1,0,Mat(),b_hist,1,&numbins,&histRange);
	calcHist(&bgr[1],1,0,Mat(),g_hist,1,&numbins,&histRange);
	calcHist(&bgr[2],1,0,Mat(),r_hist,1,&numbins,&histRange);
	//绘制直方图
	int width = 512;
	int height = 300;
	//以灰色为基底创建图像
	Mat histImage(height,width,CV_8UC3,Scalar(20,20,20));
	//从0到图像的高度归一化直方图
	normalize(b_hist,b_hist,0,height,NORM_MINMAX);
	normalize(g_hist,g_hist,0,height,NORM_MINMAX);
	normalize(r_hist,r_hist,0,height,NORM_MINMAX);
	int binStep = cvRound((float)width/(float)numbins);
	for(int i=1; i<numbins; i++){
		line(histImage,Point(binStep*(i-1),height-cvRound(b_hist.at<float>(i-1))),
			 Point(binStep*(i),height-cvRound(b_hist.at<float>(i))),
			 Scalar(255,0,0));
		line(histImage,Point(binStep*(i-1),height-cvRound(g_hist.at<float>(i-1))),
			 Point(binStep*(i),height-cvRound(g_hist.at<float>(i))),
			 Scalar(0,255,0));
		line(histImage,Point(binStep*(i-1),height-cvRound(r_hist.at<float>(i-1))),
			 Point(binStep*(i),height-cvRound(r_hist.at<float>(i))),
			 Scalar(0,0,255));
	}
	imshow("Histogram",histImage);
}

void equalizeCallback(int state,void* userData){
	Mat result;
	//rgb转YCbCr
	Mat ycrcb;
	cvtColor(m_img,ycrcb,COLOR_RGB2YCrCb);
	//图像通道分离
	vector<Mat> channels;
	split(ycrcb,channels);
	//只均衡Y通道
	equalizeHist(channels[0],channels[0]);
	//合并结果通过
	merge(channels,ycrcb);
	//将ycrcb转为rgb
	cvtColor(ycrcb,result,COLOR_YCrCb2BGR);

	imshow("equalized",result);
}

void lomoCallback(int state,void* userData){

}

void cartoonCallback(int state,void* userData){

}

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

void MyOpencvFunc::GetImageData(Mat &image, int n, int type)
{
	for(int k = 0; k < n; k++){
		//rand()Is a random number generating function
		int i = rand() % image.cols;
		int j = rand() % image.rows;
		if(type == 0){
			if(image.channels() == 1){  //gray image
				image.at<uchar>(j,i) = 255;
			}else if(image.channels() == 3) {//color image
				image.at<cv::Vec3b>(j,i)[0] = 255;
				image.at<cv::Vec3b>(j,i)[1] = 255;
				image.at<cv::Vec3b>(j,i)[2] = 255;
			}
		}else if(type == 1){
			uchar* data = image.ptr<uchar>(j);
			for(int c = 0; c < image.channels(); c++ )
				data[i*image.channels()+ c] = 255;
		}else if(type == 2){
			cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
			it += (j * image.cols + i);
			for(int c= 0; c < image.channels(); c++)
				(*it)[c] = 255;
		}
	}
}
int GetDistance(const Vec3b &color, const Vec3b &target)
{
	return abs(color[0]- target[0]) + abs(color[1]-target[1]) +
			abs(color[2]- target[2]);
}
void MyOpencvFunc::GetPixDistance(Mat &image, Mat &result)
{
	if(!image.data)
		return;
	if (image.channels() != 3)
		return;
	cv::Vec3b target(130, 190, 230);//指定颜色
	int mindistance = 100;//指定距离
	result.create(image.rows, image.cols, CV_8U);
	cv::Mat_<cv::Vec3b>::const_iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::const_iterator itend = image.end<cv::Vec3b>();
	cv::Mat_<uchar>::iterator itout = result.begin<uchar>();

	for(; it!=itend; ++it, ++itout){
		if(GetDistance(*it, target) < mindistance)
			*itout = 255;
		else *itout = 0;
	}
}

void MyOpencvFunc::ShapeImage(const Mat &image, Mat &result)
{
	result.create(image.size(), image.type());
	const int channels = image.channels() ;
	for(int j = 1; j < image.rows -1; j++){
		const uchar* previous = image.ptr<const uchar>(j-1); //up row
		const uchar* current = image.ptr<const uchar>(j);  // current row
		const uchar* next = image.ptr<const uchar>(j+1); // next row
		uchar* output = result.ptr<uchar>(j); //output row
		for(int i = channels; i < (image.cols -1) * channels; i++)
			*output++ = cv::saturate_cast<uchar>(5*current[i] -
												 current[i-channels] -
				current[i + channels] - previous[i]- next[i]);
	}
	// Set the unprocess pixels to 0
	cv::Scalar color;
	if(image.channels() == 3)
		color = cv::Scalar(0, 0, 0);
	else  color = cv::Scalar(0);
	result.row(0).setTo(color);
	result.row(result.rows - 1).setTo(color);
	result.col(0).setTo(color);
	result.col(result.cols - 1).setTo(color);
}

void MyOpencvFunc::ChangeBrightAndContrast(Mat &result, double bright,
										   double contrast, int cols)
{
	const int channels = result.channels() ;
	for(int j = 0; j < result.rows; j++){
		uchar* current = result.ptr<uchar>(j);  // current row
		for(int i = 0; i < cols * channels; i++)
			current[i] = cv::saturate_cast<uchar>(current[i]*
										(contrast *0.01)+ bright);
	}
}

void MyOpencvFunc::EqualizesHistogram(const Mat &image, Mat &result)
{
	if(image.channels() == 0){
		cv::equalizeHist(image,result);
		return;
	}
	const int channels = image.channels();
	cv::Mat* imageRGB = new cv::Mat[channels];
	split(image, imageRGB);
	for(int i=0; i< channels;++i)
		cv::equalizeHist(imageRGB[i],imageRGB[i]);
	cv::merge(imageRGB, channels, result);
	delete[] imageRGB;
}

void MyOpencvFunc::RotateImage(Mat &result, int type)
{
	if(type == 0){
		cv::Point center = cv::Point(result.cols / 2, result.rows / 2);
		double angle = -50.0;
		double scale = 0.6;
		cv::Mat rotate_matrix( 2, 3, CV_32FC1 );
		//旋转
		rotate_matrix = cv::getRotationMatrix2D(center, angle, scale);
		cv::warpAffine(result, result, rotate_matrix, result.size());
	}else if(type == 1){
		cv::Point2f srcTri[3];
		cv::Point2f dstTri[3];
		srcTri[0] = cv::Point2d(0.0, 0.0);
		srcTri[1] = cv::Point2d(2.0, 0.0);
		srcTri[2] = cv::Point2d(0.0, 1.0);
		dstTri[0] = cv::Point2d(30.0, 30.0);
		dstTri[1] = cv::Point2d(32.0, 30.0);
		dstTri[2] = cv::Point2d(30.0, 31.0);
		cv::Mat matrix( 2, 3, CV_32FC1 );
		//平移
		matrix = cv::getAffineTransform(srcTri, dstTri);
		cv::warpAffine(result, result, matrix, result.size());
	}

}

void MyOpencvFunc::TransFormImage(Mat &result, int type)
{
	//log tranform
	double c = 1.0; //尺度比例常数
	//gamma tranform
	double gamma = 0.4;  //伽马系数
	double comp = 0;     //补偿系数

	double gray = 0;
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			if(result.channels() == 1){
				gray = (double) result.at<uchar>(i, j);
				if(type == 0){
					gray =  c * log(1.0 + gray);
				}else if(type == 1){
					gray =  pow((gray + comp) / 255.0, gamma) * 255.0;
				}
				result.at<uchar>(i, j)= cv::saturate_cast<uchar>(gray);
			}
			else if(result.channels() == 3){
				for(int k = 0; k < 3; k++){
					gray = (double)result.at<cv::Vec3b>(i, j)[k];
					if(type == 0){
						gray =  c * log((double)(1.5 + gray));
					}else if(type == 1){
						gray =  pow((gray + comp) / 255.0, gamma) * 255.0;
					}
					result.at<cv::Vec3b>(i, j)[k]= cv::saturate_cast<uchar>(gray);
				}
			}
		}
	}
	//归一化处理
	cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
	cv::convertScaleAbs(result, result);
}

Mat MyOpencvFunc::regionGrowFast(const Mat &src, const Point2i seed, int throld)
{
	//convert src to gray for getting gray value of every pixel
	cv::Mat gray;
	cv::cvtColor(src,gray, cv::COLOR_RGB2GRAY);

	// set every pixel to black
	cv::Mat result = cv::Mat::zeros(src.size(), CV_8UC1);
	if((seed.x < 0) || (seed.y < 0))
		return result;
	result.at<uchar>(seed.y, seed.x) = 255;

	//grow direction sequenc
	int grow_direction[8][2] = {{-1,-1}, {0,-1}, {1,-1}, {1,0},
								{1,1}, {0,1}, {-1,1}, {-1,0}};
	//seeds collection
	std::vector<cv::Point2i> seeds;
	seeds.push_back(seed);

	//start growing
	while(! seeds.empty()){
		//get a seed
		cv::Point2i current_seed = seeds.back();
		seeds.pop_back();
		//gray value of current seed
		int seed_gray = gray.at<uchar>(current_seed.y, current_seed.x);

		for(int i = 0; i < 8; ++i){
			cv::Point2i neighbor_seed(current_seed.x + grow_direction[i][0],
					current_seed.y + grow_direction[i][1]);
			//check wether in image
			if(neighbor_seed.x < 0 || neighbor_seed.y < 0 ||
					neighbor_seed.x > (gray.cols-1) ||
					(neighbor_seed.y > gray.rows -1)){
				continue;
			}
			int value = gray.at<uchar>(neighbor_seed.y, neighbor_seed.x);
			if((result.at<uchar>(neighbor_seed.y, neighbor_seed.x) == 0) &&
					(abs(value - seed_gray) <= throld)){
				result.at<uchar>(neighbor_seed.y, neighbor_seed.x) = 255;
				seeds.push_back(neighbor_seed);
			}
		}
	}
	return result;
}

void MyOpencvFunc::baseLBP(const Mat src, Mat &dst)
{
	dst.create(src.size(), src.type());
	for(int i = 1; i < src.rows- 1; i++){
		for(int j = 1; j < src.cols- 1; j++) {
			uchar code = 0;
			uchar center = src.at<uchar>(i,j);
			code |= (src.at<uchar>(i-1,j-1) >= center) << 7;
			code |= (src.at<uchar>(i-1,j  ) >= center) << 6;
			code |= (src.at<uchar>(i-1,j+1) >= center) << 5;
			code |= (src.at<uchar>(i  ,j+1) >= center) << 4;
			code |= (src.at<uchar>(i+1,j+1) >= center) << 3;
			code |= (src.at<uchar>(i+1,j  ) >= center) << 2;
			code |= (src.at<uchar>(i+1,j-1) >= center) << 1;
			code |= (src.at<uchar>(i  ,j-1) >= center) << 0;
			dst.at<uchar>(i,j) = code;
		}
	}
}

QImage MyOpencvFunc::cvMatToQImage(const Mat &mat)
{
	// 8-bits unsigned, NO. OF CHANNELS = 1
	if(mat.type() == CV_8UC1)
	{
		QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
		// Set the color table (used to translate colour indexes to qRgb values)
		image.setColorCount(256);
		for(int i = 0; i < 256; i++)
		{
			image.setColor(i, qRgb(i, i, i));
		}
		// Copy input Mat
		uchar *pSrc = mat.data;
		for(int row = 0; row < mat.rows; row ++)
		{
			uchar *pDest = image.scanLine(row);
			memcpy(pDest, pSrc, mat.cols);
			pSrc += mat.step;
		}
		return image;
	}
	// 8-bits unsigned, NO. OF CHANNELS = 3
	else if(mat.type() == CV_8UC3)
	{
		// Copy input Mat
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	else if(mat.type() == CV_8UC4)
	{
		qDebug() << "CV_8UC4";
		// Copy input Mat
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
		return image.copy();
	}
	else
	{
		qDebug() << "ERROR: Mat could not be converted to QImage.";
		return QImage();
	}
}

Mat MyOpencvFunc::QImageTocvMat(QImage image)
{
	cv::Mat mat;
	qDebug() << image.format();
	switch(image.format())
	{
	case QImage::Format_ARGB32:
	case QImage::Format_RGB32:
	case QImage::Format_ARGB32_Premultiplied:
		mat = cv::Mat(image.height(), image.width(), CV_8UC4,
					  (void*)image.constBits(), image.bytesPerLine());
		break;
	case QImage::Format_RGB888:
		mat = cv::Mat(image.height(), image.width(), CV_8UC3,
					  (void*)image.constBits(), image.bytesPerLine());
		cv::cvtColor(mat, mat, CV_BGR2RGB);
		break;
	case QImage::Format_Indexed8:
		mat = cv::Mat(image.height(), image.width(), CV_8UC1,
					  (void*)image.constBits(), image.bytesPerLine());
		break;
	}
	return mat;
}

void MyOpencvFunc::BGRToHSI(const Mat src, Mat &dst)
{
	if(src.channels() != 3)
		return;
	dst.create(src.rows, src.cols, src.type());
	double r,g,b,h,s,i;
	for(int row = 0; row < src.rows; ++row){
		for(int col = 0; col < src.cols; ++col){
		   r = src.at<cv::Vec3b>(row, col)[0] / 255.0;
		   g = src.at<cv::Vec3b>(row, col)[1] / 255.0;
		   b = src.at<cv::Vec3b>(row, col)[2] / 255.0;
		   i = (r + b + g) / 3;

		   double min = std::min(r, std::min(b, g));
		   if(i < 0.078431)
			   s = 0.0;
		   else if(i > 0.920000)
			   s = 0.0;
		   else
			   s = 1.0 - 3.0 * min / ( r + g + b);

		  double max = std::max(r, std::max(b, g));
		  if(max == min){
			  h = 0.0;
			  s = 0.0;
		  }else {
			   h = 0.5 *(r - g + r - b) /sqrt((r - g)*( r - g) + (g -b)*(r-b));
			   if( h > 0.9999999999)
				   h = 0.0;
			   else if(h < -0.9999999999)
				   h = 180.0;
			   else
				   h =  acos(h) * 180.0 /  3.14159265358979323846;

			   if(b > g)
				   h = 360.0 - h;
		  }
		  dst.at<cv::Vec3b>(row, col)[0] = h;
		  dst.at<cv::Vec3b>(row, col)[1] = s * 255;
		  dst.at<cv::Vec3b>(row, col)[2] = i * 255;
		}
	}
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

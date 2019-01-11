#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	m_FileName = "fps.yml";
	QString filename = "./lena.jpg";
	m_image = imread(filename.toStdString());
	ShowImageMat(m_image);
	TestFunc();
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::ShowImageMat(Mat img, int posindex)
{
	QImage qimg = m_myFunc.cvMatToQImage(img);
	if(posindex == 0){
		ui->LTlabel->setPixmap(QPixmap::fromImage(qimg));
	}else if(posindex == 1){
		ui->RTlabel->setPixmap(QPixmap::fromImage(qimg));
	}
}

void MainWindow::mousePressEvent(QMouseEvent *e)
{
	if(!m_image.data)
		return;
	QPoint global_point = e->globalPos();
	QPoint point = ui->LTlabel->mapFromGlobal(global_point);
	//contentsRect表示的是QLabel的内容范围，而pixmap->rect()则表示的图像的实际大小，
	//得出图像相对于QLabel的具体偏移量，然后可以真正将鼠标坐标转换成图像上的点位置
	int xoffset = (ui->LTlabel->contentsRect().width()-
				   ui->LTlabel->pixmap()->rect().width())/2;
	int realx = point.x() - xoffset;
	if(( realx >= 0 ) && (realx <= ui->LTlabel->pixmap()->rect().width())){
		int yoffset = (ui->LTlabel->contentsRect().height()-
					   ui->LTlabel->pixmap()->height())/2;
		int realy = point.y() - yoffset;
		seed_point_ = cv::Point2i(realx, realy);
		cv::Mat result = m_image.clone();
		cv::circle(result, seed_point_, 2, cv::Scalar(0, 255, 255), 1);
		ShowImageMat(result);
	}
}

void MainWindow::on_actionWrite_triggered()
{
	/*写文件*/
	FileStorage fs(m_FileName,FileStorage::WRITE);
	int fps = 5;
	fs << "fps" << fps;

	Mat m1 = Mat::eye(2,3,CV_32F);
	Mat result = (m1+1).mul(m1+3);
	fs << "Result" << result;
	fs.release();
	std::cout << "succeed!" << std::endl;
}

void MainWindow::on_actionRead_triggered()
{
	//读文件
	FileStorage fs2(m_FileName,FileStorage::READ);
	int fps;
	fs2["fps"] >> fps;
	std::cout <<"fps:"<< fps << std::endl;

	Mat r;
	fs2["Result"] >> r;
	std::cout << "Result:"<< r << std::endl;
	fs2.release();
}

void MainWindow::on_actionShowImage_triggered()
{
	QString filepath = "./lena.jpg";
	m_myFunc.ShowImage(filepath,1);
}

void MainWindow::on_actionShowImageDeal_triggered()
{
	QString filepath = "./lena.jpg";
	m_myFunc.ShowImage(filepath,2);
}

void MainWindow::on_actionImageWithMouseEvent_triggered()
{
	QString filepath = "./lena.jpg";
	m_myFunc.ShowImageWithMouse(filepath);
}

void MainWindow::on_actionImageUsualHandle_triggered()
{
	QString filepath = "./lena.jpg";
	m_myFunc.ImageUsualHandle(filepath);
}

void MainWindow::on_actionOpenVideo_triggered()
{
	QString filepath = "./chaplin.mp4";
	m_myFunc.OpenVideo(filepath);
}

void MainWindow::GetImageData(Mat &image, int n, int type)
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

void MainWindow::GetPixDistance(Mat& image)
{
	if(!image.data)
		return;
	if (image.channels() != 3)
		return;
	cv::Mat result;
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
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

int MainWindow::GetDistance(const Vec3b &color, const Vec3b &target)
{
	return abs(color[0]- target[0]) + abs(color[1]-target[1]) +
			abs(color[2]- target[2]);
}

void MainWindow::ShapeImage(const Mat &image, Mat &result)
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

void MainWindow::ChangeBrightAndContrast(double bright, double contrast)
{
	cv::Mat result = m_image.clone();
	const int channels = result.channels() ;
	for(int j = 0; j < result.rows; j++){
		uchar* current = result.ptr<uchar>(j);  // current row
		for(int i = 0; i < m_image.cols * channels; i++)
			current[i] = cv::saturate_cast<uchar>(current[i]*
										(contrast *0.01)+ bright);
	}
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::EqualizesHistogram()
{
	cv::Mat image = m_image.clone();
	cv::Mat result = image.clone();
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
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_actionShowOnQImage_triggered()
{
	QImage img = m_myFunc.cvMatToQImage(m_image);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_actionGetPixData_triggered()
{
	cv::Mat resultRT = m_image.clone();
	GetImageData(resultRT,150,0);
	QImage imgRT = m_myFunc.cvMatToQImage(resultRT);
	ui->RTlabel->setPixmap(QPixmap::fromImage(imgRT));
	cv::Mat resultLB = m_image.clone();
	GetImageData(resultLB,150,1);
	QImage imgLB = m_myFunc.cvMatToQImage(resultLB);
	ui->LBlabel->setPixmap(QPixmap::fromImage(imgLB));
	cv::Mat resultRB = m_image.clone();
	GetImageData(resultRB,150,2);
	QImage imgRB = m_myFunc.cvMatToQImage(resultRB);
	ui->RBlabel->setPixmap(QPixmap::fromImage(imgRB));
}

void MainWindow::on_actionGetPixDistance_triggered()
{
	GetPixDistance(m_image);
}

void MainWindow::TestFunc()
{

}

void MainWindow::on_actionShapeImage_triggered()
{
	cv::Mat result;
	ShapeImage(m_image, result);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_actionFlipImage_triggered()
{
	cv::Mat result;
	m_image.copyTo(result);
	//0表示竖直翻转，1表示水平翻转
	cv::flip(result, result, 0);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_actionInsertImage_triggered()
{
	cv::Mat image = m_image.clone();
	QString filename = "./4.png";
	cv::Mat logo = cv::imread(filename.toStdString());
	if(!logo.data)
		return;
	cv::Mat result = image.clone();
	int y = (image.rows > logo.rows) ? (image.rows - logo.rows) : 0;
	int x = (image.cols > logo.cols) ? (image.cols - logo.cols) : 0;
	int height = (image.rows > logo.rows) ? logo.rows : image.rows;
	int width = (image.cols > logo.cols) ? logo.cols : image.cols;
	qDebug() << x << y << width << height;
	x = 80;
	cv::Mat imageROI =  result(cv::Rect(x,y ,width, height));
	//insert logo
	cv::addWeighted(imageROI, 1.0, logo, 0.8, 0.0, imageROI);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_horizontalSlider_bright_valueChanged(int value)
{
	double bright = ui->horizontalSlider_bright->value();
	double contrast = ui->horizontalSlider_contrast->value();
	ChangeBrightAndContrast(bright,contrast);
}

void MainWindow::on_horizontalSlider_contrast_valueChanged(int value)
{
	double bright = ui->horizontalSlider_bright->value();
	double contrast = ui->horizontalSlider_contrast->value();
	ChangeBrightAndContrast(bright,contrast);
}

void MainWindow::on_actionConvertColor_triggered()
{
	cv::Mat result;
	m_myFunc.BGRToHSI(m_image, result);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));

	cv::Mat resultRT;
	cv::cvtColor(m_image, resultRT, cv::COLOR_RGB2BGR);
	QImage imgRT = m_myFunc.cvMatToQImage(resultRT);
	ui->RTlabel->setPixmap(QPixmap::fromImage(imgRT));
	cv::Mat resultLB;
	cv::cvtColor(m_image, resultLB, cv::COLOR_RGB2GRAY);
	QImage imgLB = m_myFunc.cvMatToQImage(resultLB);
	ui->LBlabel->setPixmap(QPixmap::fromImage(imgLB));
}

void MainWindow::on_RotateBtn_clicked()
{
	/*cv::Mat result = m_image.clone();
	cv::Point center = cv::Point(result.cols / 2, result.rows / 2);
	double angle = -50.0;
	double scale = 0.6;
	cv::Mat rotate_matrix( 2, 3, CV_32FC1 );
	//旋转
	rotate_matrix = cv::getRotationMatrix2D(center, angle, scale);
	cv::warpAffine(result, result, rotate_matrix, result.size());*/

	cv::Mat result = m_image.clone();
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

	ShowImageMat(result);
}

void MainWindow::on_LogTFBtn_clicked()
{
	TransFormImage(0);
}

void MainWindow::on_GammaTFBtn_clicked()
{
	TransFormImage(1);
}

void MainWindow::TransFormImage(int type)
{
	cv::Mat result = m_image.clone();
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
	ShowImageMat(result);
}
int pyramidtype = 0;
void MainWindow::on_PyramidBtn_clicked()
{
	cv::Mat result;
	if(pyramidtype%2 == 0){
		cv::pyrUp(m_image,result, cv::Size(m_image.cols *2, m_image.rows * 2));
	}else if(pyramidtype%2 == 1){
		cv::pyrDown(m_image,result, cv::Size(m_image.cols/2, m_image.rows/2));
	}
	ShowImageMat(result);
	pyramidtype++;
}
int thresholdway = 0;
void MainWindow::on_ThresholdBtn_clicked()
{
	int numway = 5;
	cv::Mat image_gray_;
	cv::cvtColor(m_image, image_gray_, CV_BGR2GRAY);
	double thresh_ = 167;
	cv::Mat result;
	if(thresholdway%numway == 0){
		cv::threshold(image_gray_, result, thresh_,255, cv::THRESH_OTSU);
	}else if(thresholdway%numway == 1){
		cv::threshold(image_gray_, result, thresh_,255, cv::THRESH_BINARY);
	}else if(thresholdway%numway == 2){
		cv::threshold(image_gray_, result, thresh_,255, cv::THRESH_TRUNC);
	}else if(thresholdway%numway == 3){
		cv::threshold(image_gray_, result, thresh_,255, cv::THRESH_TOZERO);
	}else if(thresholdway%numway == 4){
		cv::adaptiveThreshold(image_gray_, result, 255,
							  cv::ADAPTIVE_THRESH_MEAN_C,
							  cv::THRESH_BINARY, 7, 0);
	}
	ShowImageMat(result);
	thresholdway++;
}

void MainWindow::on_SeedGrowthBtn_clicked()
{
	cv::Mat result_fast = m_myFunc.regionGrowFast(m_image, seed_point_, 4);
	ShowImageMat(result_fast,1);
}

void MainWindow::on_LBPBtn_clicked()
{
	Mat result;
	Mat gray;
	cvtColor(m_image, gray, CV_RGB2GRAY);
	m_myFunc.baseLBP(gray, result);
	ShowImageMat(result,1);
}

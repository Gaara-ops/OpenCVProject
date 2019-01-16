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

void MainWindow::EqualizesHistogram()
{
	cv::Mat image = m_image.clone();
	cv::Mat result = image.clone();
	m_myFunc.EqualizesHistogram(image,result);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_horizontalSlider_bright_valueChanged(int value)
{
	double bright = ui->horizontalSlider_bright->value();
	double contrast = ui->horizontalSlider_contrast->value();
	cv::Mat result = m_image.clone();
	m_myFunc.ChangeBrightAndContrast(result,bright,contrast,m_image.cols);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));

}

void MainWindow::on_horizontalSlider_contrast_valueChanged(int value)
{
	double bright = ui->horizontalSlider_bright->value();
	double contrast = ui->horizontalSlider_contrast->value();
	cv::Mat result = m_image.clone();
	m_myFunc.ChangeBrightAndContrast(result,bright,contrast,m_image.cols);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}
int rotetype = 0;
void MainWindow::on_RotateBtn_clicked()
{
	cv::Mat result = m_image.clone();
	if(rotetype%2 == 0){
		m_myFunc.RotateImage(result,0);
	}else if(rotetype%2 == 1){
		m_myFunc.RotateImage(result,1);
	}
	rotetype++;
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

void MainWindow::on_ConvertBtn_clicked()
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

void MainWindow::on_InsertBtn_clicked()
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

void MainWindow::on_FlipBtn_clicked()
{
	cv::Mat result;
	m_image.copyTo(result);
	//0表示竖直翻转，1表示水平翻转
	cv::flip(result, result, 0);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_ShapeBtn_clicked()
{
	cv::Mat result;
	m_myFunc.ShapeImage(m_image, result);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_GetPixDisBtn_clicked()
{
	cv::Mat result;
	m_myFunc.GetPixDistance(m_image,result);
	QImage img = m_myFunc.cvMatToQImage(result);
	ui->LTlabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_GetPixDataBtn_clicked()
{
	cv::Mat resultRT = m_image.clone();
	m_myFunc.GetImageData(resultRT,150,0);
	QImage imgRT = m_myFunc.cvMatToQImage(resultRT);
	ui->RTlabel->setPixmap(QPixmap::fromImage(imgRT));
	cv::Mat resultLB = m_image.clone();
	m_myFunc.GetImageData(resultLB,150,1);
	QImage imgLB = m_myFunc.cvMatToQImage(resultLB);
	ui->LBlabel->setPixmap(QPixmap::fromImage(imgLB));
	cv::Mat resultRB = m_image.clone();
	m_myFunc.GetImageData(resultRB,150,2);
	QImage imgRB = m_myFunc.cvMatToQImage(resultRB);
	ui->RBlabel->setPixmap(QPixmap::fromImage(imgRB));
}
int transformtype = 0;
void MainWindow::on_TransFormBtn_clicked()
{
	cv::Mat result = m_image.clone();
	if(transformtype%2 == 0){
		m_myFunc.TransFormImage(result,0);
	}else if(transformtype%2 == 1){
		m_myFunc.TransFormImage(result,1);
	}
	transformtype++;
	ShowImageMat(result);
}

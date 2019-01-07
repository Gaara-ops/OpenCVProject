#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	m_FileName = "fps.yml";

}

MainWindow::~MainWindow()
{
	delete ui;
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

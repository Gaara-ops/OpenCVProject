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

private slots:
	//写入数据到yml文件
	void on_actionWrite_triggered();
	//从yml文件读取数据
	void on_actionRead_triggered();
	//显示一张图片
	void on_actionShowImage_triggered();

	void on_actionShowImageDeal_triggered();

	void on_actionImageWithMouseEvent_triggered();

	void on_actionImageUsualHandle_triggered();

	void on_actionOpenVideo_triggered();

private:
	Ui::MainWindow *ui;
	String m_FileName;
	MyOpencvFunc  m_myFunc;
};

#endif // MAINWINDOW_H

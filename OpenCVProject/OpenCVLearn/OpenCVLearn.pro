#-------------------------------------------------
#
# Project created by QtCreator 2018-12-26T09:15:52
#
#-------------------------------------------------
QT       += core gui

#引入c++11
CONFIG  += C++11

OpenCV_DIR = F:/opencv/opencv-libreleaseqt
#OpenCV_DIR = F:/opencv3.1/opencv-libdebug
#引入头文件的路径
INCLUDEPATH += $${OpenCV_DIR}/include
#引入路径下的所有库
LIBS += $${OpenCV_DIR}/x86/mingw/lib/libopencv_*

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = OpenCVLearn
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    myopencvfunc.cpp

HEADERS  += mainwindow.h \
    myopencvfunc.h \
    myhead.h

FORMS    += mainwindow.ui

#-------------------------------------------------
#
# Project created by QtCreator 2014-03-18T11:14:50
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = PCA
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    functions.cpp


QMAKE_CXXFLAGS += -std=c++11
CONFIG += link_pkgconfig
PKGCONFIG += opencv
INCLUDEPATH += /home/ramiro/flandmark-master/libflandmark/
LIBS += /home/ramiro/flandmark-master/build/libflandmark/libflandmark_static.a

HEADERS += \
    functions.h

#-------------------------------------------------
#
# Project created by QtCreator 2013-09-11T10:49:05
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = VideoFeaturesExtraction
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    functions.cpp

HEADERS += \
    functions.h

CONFIG += link_pkgconfig
PKGCONFIG += opencv
INCLUDEPATH += /home/ramiro/flandmark-master/libflandmark/
LIBS += /home/ramiro/flandmark-master/build/libflandmark/libflandmark_static.a

#ifndef SETTINGSBOX_H
#define SETTINGSBOX_H

#include <QtGui/QWidget>

#include <map>
#include <sstream>

using namespace std;

namespace Ui
{
    class SettingsBox;
}

class SettingsBox : public QWidget
{
    Q_OBJECT

public:
    SettingsBox();
    ~SettingsBox();

private:
    Ui::SettingsBox *ui;
    map< QString,pair<int,int> > points;
    int maxPointNumber;

private slots:
    void on_pointDelete_clicked();
    void on_pointNew_clicked();
    void on_pointYValue_textChanged(QString );
    void on_pointXValue_textChanged(QString );
    void on_schemeUse_clicked();
    void on_actionExit_triggered();
    void on_pointBox_currentIndexChanged(int index);
    void on_functionBox_currentIndexChanged(int index);
    void on_functionDelete_clicked();
    void on_functionName_textChanged(QString );
    void on_functionNew_clicked();
};


template<typename From, typename To>
static To convert(const From &s, bool playerEntry = false)
{
    stringstream ss;
    To d;
    ss << s;
    if(ss >> d)
        return d;
    /*
    if(!playerEntry)
    {
        cerr << endl
             << "error: conversion failed, used default" << endl;
    }
    */
    return NULL;
}

#endif // SETTINGSBOX_H

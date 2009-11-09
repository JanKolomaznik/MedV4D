#ifndef SETTINGSBOX_H
#define SETTINGSBOX_H

#include <QtGui/QWidget>

#include <map>

#include "TF/TFScheme.h"
#include "TF/Convert.h"

using namespace std;

namespace Ui{

    class SettingsBox;
}

class SettingsBox : public QWidget{

    Q_OBJECT

public:
    SettingsBox();
    ~SettingsBox();

private:
    Ui::SettingsBox *ui;

	TFScheme* currentScheme;
	TFFunction* currentFunction;
	TFPoint* currentPoint;

private slots:
    void on_pointDelete_clicked();
    void on_pointSave_clicked();
    void on_schemeUse_clicked();
    void on_actionExit_triggered();
    void on_pointBox_currentIndexChanged(int index);
    void on_functionBox_currentIndexChanged(int index);
    void on_functionDelete_clicked();
    void on_functionSave_clicked();
};

#endif SETTINGSBOX_H
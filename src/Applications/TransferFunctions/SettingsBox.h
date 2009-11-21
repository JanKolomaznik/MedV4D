#ifndef SETTINGSBOX
#define SETTINGSBOX

#include <QtGui/QWidget>

#include <map>

#include <TF/TFScheme.h>
#include <TF/Convert.h>
#include <TF/TFPaintingWidget.h>

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
    Ui::SettingsBox* ui;

	TFScheme* savedFunctions;
	TFFunction* currentFunction;

	PaintingWidget* painter;

private slots:
    void on_schemeUse_clicked();
    void on_actionExit_triggered();
    void on_functionBox_currentIndexChanged(int index);
    void on_functionDelete_clicked();
    void on_functionSave_clicked();
};

#endif //SETTINGSBOX
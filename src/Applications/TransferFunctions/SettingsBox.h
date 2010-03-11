#ifndef SETTINGSBOX
#define SETTINGSBOX

#include <map>

#include <TF/TFAFunction.h>
#include <TF/Convert.h>
#include <TF/TFXmlReader.h>
#include <TF/TFXmlWriter.h>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

using namespace std;

namespace Ui{

    class SettingsBox;
}

class SettingsBox : public QWidget{

    Q_OBJECT

public:
	SettingsBox();
    ~SettingsBox();

	virtual void build();

private:
    Ui::SettingsBox* ui;

protected:
	TFAFunction* tf;
	virtual TFAFunction* createDefaultTransferFunction();
	virtual void setupToolsAndPainter();

	QWidget* toolsWidget;
	QWidget* painterWidget;

protected slots:
    void on_schemeUse_clicked();

    void on_actionExit_triggered();
    void on_saveScheme_triggered();	//TODO abstract
    void on_loadScheme_triggered();	//TODO abstract

signals:
	void UseTransferFunction(TFAFunction* transferFunction);
};

#endif //SETTINGSBOX
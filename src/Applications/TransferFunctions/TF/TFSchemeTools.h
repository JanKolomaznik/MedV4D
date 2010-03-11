#ifndef TF_TOOLSWIDGET
#define TF_TOOLSWIDGET

#include <QtGui/QWidget>
#include <TF/TFScheme.h>

namespace Ui{

	class TFSchemeTools;
}

class TFSchemeTools: public QWidget{

	Q_OBJECT

public:

	TFSchemeTools();

	~TFSchemeTools();

	void setScheme(TFScheme* scheme);

	void save();

	void load();

private slots:

    void on_functionBox_currentIndexChanged(int index);
    void on_functionDelete_clicked();
    void on_functionSave_clicked();

private:
	Ui::TFSchemeTools* tools;

	TFScheme* currentScheme;

signals:
	void CurrentFunctionChanged();
};

#endif //TF_TOOLSWIDGET
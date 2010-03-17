#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <TFAlgorithms.h>

using namespace std;

namespace Ui{

	class TFWindow;
}

class TFWindow : public QWidget{

    Q_OBJECT

public:
	TFWindow();
    ~TFWindow();

	virtual void build();

//protected:
/*
	TFAFunction* tf;
	virtual TFAFunction* createDefaultTransferFunction();
	virtual void setupToolsAndPainter();

	QWidget* toolsWidget;
	QWidget* painterWidget;
*/
protected slots:
    //void on_schemeUse_clicked();

    void on_exit_triggered();
    void on_save_triggered();
    void on_load_triggered();

	void on_simple_triggered();

	void modify_data(TFAbstractFunction &transferFunction);

signals:
	void AdjustByTransferFunction(TFAbstractFunction &transferFunction);

private:	
    Ui::TFWindow* ui;

	TFAbstractHolder* _holder;

	void setupHolder();
};

#endif //TF_WINDOW
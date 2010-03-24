#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <TFAlgorithms.h>

#define MENU_SPACE 30

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
signals:
	void AdjustByTransferFunction(TFAbstractFunction &transferFunction);
	void ResizeHolder(const QRect rect);

protected slots:
    //void on_schemeUse_clicked();

    void on_exit_triggered();
    void on_save_triggered();
    void on_load_triggered();

	void on_newTF_triggered(TFType &tfType);

	void modify_data(TFAbstractFunction &transferFunction);

	void resizeEvent(QResizeEvent *event);

private:	
    Ui::TFWindow* ui;

	TFAbstractHolder* _holder;
	TFActions tfActions;

	void setupHolder();
};

#endif //TF_WINDOW
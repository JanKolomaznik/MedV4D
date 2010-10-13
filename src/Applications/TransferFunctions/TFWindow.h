#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <TFHolderFactory.h>

#define MENU_SPACE 30

namespace Ui{

	class TFWindow;
}

class TFWindow : public TFWindowI{

    Q_OBJECT

public:
	TFWindow();
    ~TFWindow();

	void build();

	void modifyData(TFAbstractFunction &transferFunction);
	void getHistogram();

signals:
	void ResizeHolder(const QRect rect);
	void AdjustByTransferFunction(TFAbstractFunction &transferFunction);
	void HistogramRequest();

public slots:
	void receive_histogram(const TFHistogram& histogram);

protected slots:
    void on_exit_triggered();
    void on_save_triggered();
    void on_load_triggered();
	void newTF_triggered(TFType &tfType);

	void on_actionTest_triggered();

protected:
	void resizeEvent(QResizeEvent* e);

private:	
    Ui::TFWindow* ui_;

	TFAbstractHolder* holder_;
	TFActions tfActions_;

	void setupHolder();
};

#endif //TF_WINDOW
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

class TFWindow : public QWidget{

    Q_OBJECT

public:
	TFWindow();
    ~TFWindow();

	void build();

signals:
	void AdjustByTransferFunction(TFAbstractFunction &transferFunction);
	void ResizeHolder(const QRect rect);

protected slots:
    void on_exit_triggered();
    void on_save_triggered();
    void on_load_triggered();

	void newTF_triggered(TFType &tfType);

	void modify_data(TFAbstractFunction &transferFunction);

protected:
	void resizeEvent(QResizeEvent* e);

private:	
    Ui::TFWindow* ui_;

	TFAbstractHolder* holder_;
	TFActions tfActions_;

	void setupHolder();
};

#endif //TF_WINDOW
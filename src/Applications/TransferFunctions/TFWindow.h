#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QMenuBar>

#include <TFHolderFactory.h>
#include <ui_TFWindow.h>

namespace M4D {
namespace GUI {

class TFWindow : public QWidget{

    Q_OBJECT

public:
	TFWindow();
    ~TFWindow();

	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		if(!holder_) 
		{
			QMessageBox::critical(this, QObject::tr("Transfer Functions"),
				QObject::tr("No function available!"));
			return false;
		}
		return holder_->applyTransferFunction<ElementIterator>(begin, end);
	}

	void createMenu(QMenuBar* menubar);

	void setupDefault();

signals:
	void ResizeHolder(const QRect rect);

protected slots:
    void on_exit_triggered();
    void on_save_triggered();
    void on_load_triggered();
	void newTF_triggered(TFHolderType &tfType);

protected:
	void resizeEvent(QResizeEvent* e);

private:	
    Ui::TFWindow* ui_;

	QMenu* menuTF_;
	QMenu* menuNew_;
    QAction *actionLoad_;
    QAction *actionSave_;
    QAction *actionExit_;

	TFAbstractHolder* holder_;
	TFActions tfActions_;

	void setupHolder();
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW
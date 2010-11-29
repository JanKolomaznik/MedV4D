#ifndef TF_DOCKHOLDER
#define TF_DOCKHOLDER

#include <QtGui/QDockWidget>

#include <TFTypes.h>
#include <TFAbstractHolder.h>

#include <ui_TFDockHolder.h>

namespace M4D {
namespace GUI {

class TFDockHolder : public QDockWidget, public Ui::TFDockHolder{

    Q_OBJECT

public:

	TFDockHolder(const QString& title, QWidget* tfWindow, TFSize index);
    ~TFDockHolder();

	bool connectSignals();

signals:

	void CloseDockHolder();
	void ResizeHolder(const TFSize& index, const QRect&);

protected:

	void closeEvent(QCloseEvent*);
	//void resizeEvent(QResizeEvent*);

private:

	TFSize index_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_DOCKHOLDER
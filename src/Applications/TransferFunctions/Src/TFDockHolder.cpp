#include <TFDockHolder.h>

#include <QtGui/QResizeEvent>

namespace M4D {
namespace GUI {

TFDockHolder::TFDockHolder(const QString& title, QWidget* tfWindow, TFSize index):
	QDockWidget(title, tfWindow),
	index_(index){

	setupUi(this);
	setFocusPolicy(Qt::StrongFocus);
}

TFDockHolder::~TFDockHolder(){}

bool TFDockHolder::connectSignals(){

	bool closeConnected = QObject::connect(this, SIGNAL( CloseDockHolder() ), parent(), SLOT( close_triggered() ));
	tfAssert(closeConnected);
	bool resizeConnected = QObject::connect(this, SIGNAL( ResizeHolder(const TFSize&, const QRect&) ), widget(), SLOT( size_changed(const TFSize&, const QRect&) ));
	tfAssert(resizeConnected);

	return closeConnected &&
		resizeConnected;
}

void TFDockHolder::closeEvent(QCloseEvent*){

	emit CloseDockHolder();
}
/*
void TFDockHolder::resizeEvent(QResizeEvent*){

	QRect holderRect = rect();
	emit ResizeHolder(index_, holderRect);
}
*/
} // namespace GUI
} // namespace M4D
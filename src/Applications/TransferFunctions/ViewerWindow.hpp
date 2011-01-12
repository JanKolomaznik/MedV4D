#ifndef VIEWER_WINDOW
#define VIEWER_WINDOW

#include <QtGui>
#include <QtCore>
#include "ui_ViewerWindow.h"
#include "GUI/utils/TransferFunctionBuffer.h"
#include "Imaging/Histogram.h"
#include <TFPalette.h>

class ViewerWindow: public QMainWindow, public Ui::ViewerWindow
{
	Q_OBJECT;

	typedef M4D::GUI::TransferFunctionBuffer1D Buffer1D;
	typedef M4D::GUI::TransferFunctionBuffer1D::MappedInterval Interval;

public:

	ViewerWindow();

	~ViewerWindow();

public slots:

	void applyTransferFunction();

	void openFile();

	void openFile( const QString &aPath );

	void updateTransferFunction();

	void toggleInteractiveTransferFunction( bool aChecked );

	void updateToolbars();

	void changeViewerType( int aRendererType );

	void changeColorMapType( int aColorMap );

protected:

	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mProdconn;
	M4D::GUI::TFPalette *mTransferFunctionEditor;

	QTimer	mTransFuncTimer;
	M4D::Common::TimeStamp mLastTimeStamp;

private:

	M4D::Imaging::Histogram32::Ptr histogram_;

	void sendImageHistogram(M4D::Imaging::AImage::Ptr image);

};


#endif //VIEWER_WINDOW

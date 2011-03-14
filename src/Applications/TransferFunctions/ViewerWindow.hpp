#ifndef VIEWER_WINDOW
#define VIEWER_WINDOW

#include "ui_ViewerWindow.h"

#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/utils/ImageDataRenderer.h"
#include "Imaging/ImageTools.h"
#include "Imaging/Histogram.h"

#include <cmath>

#include <QtGui>
#include <QtCore>

#include <TFPalette.h>

class ViewerWindow: public QMainWindow, public Ui::ViewerWindow{

	Q_OBJECT

	typedef M4D::GUI::TransferFunctionBuffer1D Buffer1D;
	typedef M4D::GUI::TransferFunctionBuffer1D::MappedInterval Interval;

public:

	ViewerWindow();

	~ViewerWindow();

public slots:

	void openFile();
	void  openFile( const QString &aPath );

	void applyTransferFunction();
	void toggleInteractiveTransferFunction( bool aChecked );

	void changeViewerType( int aRendererType );
	void changeColorMapType( int aColorMap );

	void updateTransferFunction();
	void updateToolbars();

protected:

	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mProdconn;
	M4D::GUI::TFPalette *mTransferFunctionEditor;

	QTimer	mTransFuncTimer;
	M4D::Common::TimeStamp mLastTimeStamp;

};


#endif //VIEWER_WINDOW

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
#include <QtGui\QMessageBox>

#include <TFPalette.h>
#include <TFFunctionInterface.h>

class ViewerWindow: public QMainWindow, public Ui::ViewerWindow{

	Q_OBJECT

public:

	ViewerWindow();

	~ViewerWindow();

public slots:

	void openFile();
	void openFile( const QString &aPath );

	void applyTransferFunction();
	void toggleInteractiveTransferFunction( bool aChecked );

	void changeViewerType( int aRendererType );
	void changeColorMapType( int aColorMap );

	void updateTransferFunction();
	void updateToolbars();

	void updatePreview(M4D::GUI::TF::Size index);

protected:

	typedef M4D::GUI::TransferFunctionBuffer1D Buffer1D;
	typedef Buffer1D::MappedInterval Interval;

	typedef M4D::Imaging::Histogram64 Histogram;
	typedef M4D::GUI::TF::Histogram TFHistogram;

	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mProdconn;
	M4D::GUI::TFPalette::Ptr editingSystem_;

	Buffer1D::Ptr buffer_;

	QTimer	changeChecker_;
	M4D::Common::TimeStamp lastChange_;

	bool fileLoaded_;

	bool fillBufferFromTF_(M4D::GUI::TFFunctionInterface::Const function, Buffer1D::Ptr& buffer);

	void closeEvent(QCloseEvent*);
};


#endif //VIEWER_WINDOW

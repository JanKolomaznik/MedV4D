#ifndef VIEWER_WINDOW
#define VIEWER_WINDOW

#include <QtGui>
#include <QtCore>
#include "ui_ViewerWindow.h"
#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/widgets/TransferFunction1DEditor.h"

class ViewerWindow: public QMainWindow, public Ui::ViewerWindow
{
	Q_OBJECT;
public:
	ViewerWindow();

	~ViewerWindow();


public slots:

	void
	applyTransferFunction();

	void
	openFile();

	void 
	openFile( const QString &aPath );

	void
	updateTransferFunction();

	void
	toggleInteractiveTransferFunction( bool aChecked );

	void
	updateToolbars();

	void
	changeViewerType( int aRendererType );

	void
	changeColorMapType( int aColorMap );

	void
	testSlot();

	void
	initAfterLoopStart();
signals:
	void
	callInitAfterLoopStart();
protected:
	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mProdconn;
	M4D::GUI::TransferFunction1DEditor *mTransferFunctionEditor;

	QTimer	mTransFuncTimer;
	M4D::Common::TimeStamp mLastTimeStamp;

	QComboBox *mColorTransformChooser;

	M4D::GUI::Viewer::ViewerController::Ptr mViewerController;
private:

};


#endif /*VIEWER_WINDOW*/

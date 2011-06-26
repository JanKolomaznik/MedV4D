#ifndef VIEWER_WINDOW
#define VIEWER_WINDOW

#include <QtGui>
#include <QtCore>
#include "ui_ViewerWindow.h"
#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/widgets/TransferFunction1DEditor.h"
#include "AnnotationEditorController.hpp"
#include "GUI/widgets/ProgressInfoDialog.h"
#include "backendForDICOM/DcmProvider.h"
#ifdef OIS_ENABLED
#include "JoyInput.h"
#endif


class ViewerWindow: public QMainWindow, public Ui::ViewerWindow
{
	Q_OBJECT;
public:
	ViewerWindow();

	~ViewerWindow();

	M4D::GUI::Viewer::GeneralViewer *
	getSelectedViewer();
	
public slots:

	void
	applyTransferFunction();

	void
	openFile();

	void 
	openFile( const QString &aPath );

	void 
	openDicom( const QString &aPath );

	void
	updateTransferFunction();

	void
	toggleInteractiveTransferFunction( bool aChecked );

	void
	updateToolbars();

	void
	changeViewerType( int aRendererType );

	void
	changeColorMapType( const QString & aColorMapName );

	void
	testSlot();

	void
	initAfterLoopStart();

	void
	dataLoaded();

#ifdef OIS_ENABLED
	void
	updateJoyControl();
#endif

signals:
	void
	callInitAfterLoopStart();
protected:
	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mProdconn;
	M4D::GUI::TransferFunction1DEditor *mTransferFunctionEditor;

	QTimer	mTransFuncTimer;
	M4D::Common::TimeStamp mLastTimeStamp;

	QComboBox *mColorTransformChooser;

	AnnotationEditorController::Ptr mViewerController;
	QMainWindow *mMainWin2;

	ProgressInfoDialog::Ptr mProgressDialog;
	M4D::Dicom::DicomObjSetPtr mDicomObjSet;

#ifdef OIS_ENABLED
	// OIS
	JoyInput mJoyInput;
	QTimer mJoyTimer;
#endif 

private:

};


#endif /*VIEWER_WINDOW*/

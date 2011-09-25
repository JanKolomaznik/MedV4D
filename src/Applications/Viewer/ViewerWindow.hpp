#ifndef VIEWER_WINDOW
#define VIEWER_WINDOW

#include <QtGui>
#include <QtCore>
#include "ui_ViewerWindow.h"
#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/utils/ProxyViewerController.h"
#include "GUI/utils/ProxyRenderingExtension.h"
#include "GUI/widgets/TransferFunction1DEditor.h"
#include "GUI/widgets/ProgressInfoDialog.h"
#include "GUI/widgets/MainWindow.h"

#include "backendForDICOM/DcmProvider.h"
#ifdef OIS_ENABLED
#include "JoyInput.h"
#endif

#include "GUI/TF/TFPalette.h"
#include "GUI/TF/TFFunctionInterface.h"

class ViewerWindow: public M4D::GUI::MainWindow, public Ui::ViewerWindow
{
	Q_OBJECT;
public:
	ViewerWindow();

	~ViewerWindow();

	M4D::GUI::Viewer::GeneralViewer *
	getSelectedViewer();

	void
	addRenderingExtension( M4D::GUI::Viewer::RenderingExtension::Ptr aRenderingExtension );

	void
	setViewerController( M4D::GUI::Viewer::AViewerController::Ptr aViewerController );
	
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
	updateGui();

	void
	testSlot();

	void
	initAfterLoopStart();

	void
	dataLoaded();

	void
	selectedViewerSettingsChanged();

	void
	showSettingsDialog();

#ifdef OIS_ENABLED
	void
	updateJoyControl();
#endif
	void
	updateInfoInStatusBar( const QString &aInfo );
signals:
	void
	callInitAfterLoopStart();
protected:
	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mProdconn;
	M4D::GUI::TransferFunction1DEditor *mTransferFunctionEditor;

	QTimer	mTransFuncTimer;
	M4D::Common::TimeStamp mLastTimeStamp;

	QComboBox *mColorTransformChooser;

	ProxyViewerController::Ptr mViewerController;
	ProxyRenderingExtension::Ptr mRenderingExtension;
	QMainWindow *mMainWin2;
	
	QLabel *mInfoLabel;


	ProgressInfoDialog::Ptr mProgressDialog;
	M4D::Dicom::DicomObjSetPtr mDicomObjSet;

#ifdef OIS_ENABLED
	// OIS
	JoyInput mJoyInput;
	QTimer mJoyTimer;
#endif 

	M4D::GUI::TFPalette::Ptr mTFEditingSystem;

private:

};


#endif /*VIEWER_WINDOW*/

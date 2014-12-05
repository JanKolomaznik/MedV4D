#ifndef VIEWER_WINDOW
#define VIEWER_WINDOW

#include <QtGui>
#include <QtCore>
#include "ui_ViewerWindow.h"
#include <vorgl/TransferFunctionBuffer.hpp>
#include "MedV4D/GUI/utils/ProxyViewerController.h"
#include "MedV4D/GUI/utils/ProxyRenderingExtension.h"
#include "MedV4D/GUI/managers/DatasetManager.h"
#include "MedV4D/GUI/widgets/TransferFunction1DEditor.h"
#include "MedV4D/GUI/widgets/ProgressInfoDialog.h"
#include "MedV4D/GUI/widgets/MainWindow.h"
#include "MedV4D/GUI/managers/OpenGLManager.h"

#include "MedV4D/DICOMInterface/DcmProvider.h"

#ifdef OIS_ENABLED
#include "JoyInput.h"
#endif

#include "MedV4D/GUI/TF/Palette.h"
#include "MedV4D/GUI/TF/FunctionInterface.h"

#include "tfw/PaletteWidget.hpp"
#include "tfw/data/ATransferFunction.hpp"
#include "tfw/data/TransferFunctionLoading.hpp"
#include "tfw/data/TransferFunctionPalette.hpp"

#include "DatasetManager.hpp"


typedef std::list< M4D::GUI::Viewer::GeneralViewer * > ViewerList;
struct TransferFunctionBufferUsageRecord
{
	vorgl::TransferFunctionBufferInfo info;
	ViewerList viewers;
};
typedef std::map< M4D::Common::IDNumber, TransferFunctionBufferUsageRecord > TransferBufferUsageMap;




class ViewerWindow: public M4D::GUI::MainWindow, public Ui::ViewerWindow
{
	Q_OBJECT;
public:
	ViewerWindow();

	~ViewerWindow();

	void initialize();

	M4D::GUI::Viewer::GeneralViewer *
	getSelectedViewer();

	void
	addRenderingExtension( M4D::GUI::Viewer::RenderingExtension::Ptr aRenderingExtension );

	void
	setViewerController( M4D::GUI::Viewer::AViewerController::Ptr aViewerController );

	void
	dataLoaded(DatasetManager::DatasetID aId);

	void
	processModule(AModule &aModule);

public slots:

	void
	applyTransferFunction();

	void
	openFile();

	void
	closeAllFiles();

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
	selectedViewerSettingsChanged();

	void
	changedViewerSelection();

	void
	transferFunctionAdded( int idx );

	void
	changedTransferFunctionSelection();

	void
	transferFunctionModified( int idx );

	void
	showSettingsDialog();

#ifdef OIS_ENABLED
	void
	updateJoyControl();
#endif
	void
	updateInfoInStatusBar( const QString &aInfo );

	void
	computeHistogram(DatasetManager::DatasetID aId/*M4D::Imaging::AImage::Ptr aImage*/);

	//TMP
	void
	denoiseImage();
signals:
	void
	callInitAfterLoopStart();
protected:

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
	M4D::DatasetID mDatasetId;

#ifdef OIS_ENABLED
	// OIS
	JoyInput mJoyInput;
	QTimer mJoyTimer;
#endif

	//M4D::GUI::Palette::Ptr mTFEditingSystem;
	TransferBufferUsageMap mTFUsageMap;

	QFileDialog	*mOpenDialog;

	std::shared_ptr<tfw::TransferFunctionPalette> mTFPalette;
	std::unique_ptr<tfw::PaletteWidget> mTFPaletteWidget;

	DatasetManager mDatasetManager;

private:

};


#endif /*VIEWER_WINDOW*/

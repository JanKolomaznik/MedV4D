#include "ViewerWindow.hpp"
//#include "MedV4D/GUI/utils/ImageDataRenderer.h"
#include "MedV4D/Imaging/ImageTools.h"
#include "MedV4D/Imaging/Histogram.h"
#include <cmath>

#include "MedV4D/GUI/widgets/PythonTerminal.h"
#include "MedV4D/GUI/widgets/MultiDockWidget.h"
#include "MedV4D/GUI/managers/ViewerManager.h"
#include "MedV4D/GUI/utils/ProxyRenderingExtension.h"
#include "MedV4D/GUI/utils/ViewerAction.h"
#include "MedV4D/GUI/utils/QtM4DTools.h"
#include "MedV4D/GUI/widgets/SettingsDialog.h"
#include "MedV4D/GUI/widgets/ViewerControls.h"
#include "ExtendedViewerControls.hpp"
#include "DatasetManagerWidget.hpp"
#include "MedV4D/Common/MathTools.h"
#include <boost/thread.hpp>

#include "Statistics.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>

#include "ViewerModule.hpp"

#include <boost/scope_exit.hpp>

#ifdef USE_CUDA
#include "MedV4D/Imaging/cuda/MedianFilter.h"
#endif

#include <QtGui>
namespace M4D
{
namespace GUI
{
namespace Viewer
{

class GeneralViewerFactory: public AViewerFactory
{
public:
	GeneralViewerFactory()
	{}
	typedef std::shared_ptr< GeneralViewerFactory > Ptr;

	AGLViewer *
	createViewer()
	{
		GeneralViewer *viewer = new GeneralViewer();
		ViewerManager::getInstance()->registerViewer(viewer);
		//TODO - better id assignment, fix - AViewer not a base class
		viewer->setName(std::string("Viewer ") + std::to_string(++mViewerCounter));

		if( mRenderingExtension ) {
			viewer->setRenderingExtension( mRenderingExtension );
		}
		if( mViewerController ) {
			viewer->setViewerController( mViewerController );
		}
		viewer->setLUTWindow(glm::fvec2(1500.0f,100.0f));

		viewer->enableShading( GET_SETTINGS( "gui.viewer.volume_rendering.shading_enabled", bool, true ) );

		viewer->enableJittering( GET_SETTINGS( "gui.viewer.volume_rendering.jittering_enabled", bool, true ) );

		viewer->setJitterStrength( GET_SETTINGS( "gui.viewer.volume_rendering.jitter_strength", double, 1.0 ) );

		viewer->enableIntegratedTransferFunction( GET_SETTINGS( "gui.viewer.volume_rendering.integrated_transfer_function_enabled", bool, true ) );

		viewer->setRenderingQuality( GET_SETTINGS( "gui.viewer.volume_rendering.rendering_quality", int, qmNormal ) );

		viewer->enableBoundingBox( GET_SETTINGS( "gui.viewer.volume_rendering.bounding_box_enabled", bool, true ) );

		Vector4d color = GET_SETTINGS( "gui.viewer.background_color", Vector4d, Vector4d( 0.0, 0.0, 0.0, 1.0 ) );
		//Vector4d color = GET_SETTINGS( "gui.viewer.background_color", Vector4d, Vector4d( 1.0, 1.0, 1.0, 1.0 ) );
		viewer->setBackgroundColor( QColor::fromRgbF( color[0], color[1], color[2], color[3] ) );
		//viewer->setBackgroundColor(QColor( 255, 255, 255));
		return viewer;
	}
	void
	setRenderingExtension( RenderingExtension::Ptr aRenderingExtension )
	{
		mRenderingExtension = aRenderingExtension;
	}

	void
	setViewerController( AViewerController::Ptr aController )
	{
		mViewerController = aController;
	}
protected:
	RenderingExtension::Ptr mRenderingExtension;
	AViewerController::Ptr	mViewerController;

	int mViewerCounter = 0;
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


ViewerWindow::ViewerWindow()
{
	setupUi( this );
	setAnimated( false );

	#ifdef WIN32
		//Reposition console window
		QRect myRegion=frameGeometry();
		QPoint putAt=myRegion.topRight();
		SetWindowPos(GetConsoleWindow(),(HWND)winId(),putAt.x()+1,putAt.y(),0,0,SWP_NOSIZE);
	#endif
}

void
ViewerWindow::initialize()
{
	boost::filesystem::path dataDirName = GET_SETTINGS( "application.data_directory", std::string, (boost::filesystem::current_path() / "data").string() );
	M4D::GUI::Renderer::gSliceRendererShaderPath = dataDirName / "shaders";
	M4D::GUI::Renderer::gVolumeRendererShaderPath = dataDirName / "shaders";
	//M4D::gPickingShaderPath = dataDirName / "shaders" / "PickingShader.cgfx";

	mViewerController = ProxyViewerController::Ptr( new ProxyViewerController );
	mRenderingExtension = ProxyRenderingExtension::Ptr( new ProxyRenderingExtension );
//***********************************************************
	/*std::vector<M4D::GUI::TF::Size> dataCT1D(1, 4096);	//default CT
	mTFEditingSystem = M4D::GUI::Palette::Ptr(new M4D::GUI::Palette(this, dataCT1D));
	mTFEditingSystem->setupDefault();*/

	//createDockWidget( tr("Transfer Function Palette Old"), Qt::RightDockWidgetArea, mTFEditingSystem.get(), true );
	/*QDockWidget* dockWidget = new QDockWidget("Transfer Function Palette", this);

	dockWidget->setWidget( &(*mTFEditingSystem) );
	dockWidget->setFeatures(QDockWidget::AllDockWidgetFeatures);
	dockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);

	addDockWidget(Qt::LeftDockWidgetArea, dockWidget);*/
	//dockWidget->setFloating(true);

	//loadAllSavedTFEditorsIntoPalette( *mTFEditingSystem, GET_SETTINGS( "gui.transfer_functions.load_path", std::string, std::string( "./data/TF" ) ) );

	//LOG( "TF framework initialized" );

	mTFPaletteWidget = std::unique_ptr<tfw::PaletteWidget>(new tfw::PaletteWidget(this));
	mTFPalette = std::make_shared<tfw::TransferFunctionPalette>();
	tfw::fillTFPalette(*mTFPalette, GET_SETTINGS( "gui.transfer_functions.load_path", std::string, std::string("./data/TF")));
	mTFPaletteWidget->setPalette(mTFPalette);
	createDockWidget( tr("Transfer Function Palette New"), Qt::RightDockWidgetArea, mTFPaletteWidget.get(), false );

	mTFPaletteWidget->setWrapEditorCallback(
			[this](tfw::ATransferFunctionEditor *aEditor) -> QWidget * {
				return createDockWidget(aEditor->tfName(), Qt::RightDockWidgetArea, aEditor, true );
			});
#ifdef USE_PYTHON
	M4D::GUI::TerminalWidget *mTerminal = new M4D::GUI::PythonTerminal;
	createDockWidget( tr("Python Terminal" ), Qt::BottomDockWidgetArea, mTerminal, false );
	LOG( "Python terminal initialized" );
#endif //USE_PYTHON

	ExtendedViewerControls *mViewerControls = new ExtendedViewerControls(mDatasetManager);
	createDockWidget( tr("Viewer Controls" ), Qt::RightDockWidgetArea, mViewerControls, false );
	QObject::connect(
			M4D::ApplicationManager::getInstance(),
			&M4D::ApplicationManager::viewerSelectionChanged,
			[mViewerControls] () {
				auto viewer = ViewerManager::getInstance()->getSelectedViewer();
				auto genViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (viewer);
				mViewerControls->setViewer(genViewer);
			});
	QObject::connect( M4D::ApplicationManager::getInstance(), SIGNAL( selectedViewerSettingsChanged() ), mViewerControls, SLOT( updateControls() ) );
	LOG( "Viewer controls GUI initialized" );

	auto datasetManagerWidget = new DatasetManagerWidget(mDatasetManager);
	createDockWidget( tr("Datasets" ), Qt::RightDockWidgetArea, datasetManagerWidget, false );
	//************* TOOLBAR & MENU *****************
	ViewerActionSet &actions = ViewerManager::getInstance()->getViewerActionSet();
	QToolBar *toolbar = createToolBarFromViewerActionSet( actions, "Viewer settings" );
	addToolBar( toolbar );

	QAction *action = new QAction( "Denoise", this );
	QObject::connect( action, SIGNAL( triggered(bool) ), this, SLOT( denoiseImage() ) );
	toolbar = new QToolBar( "Image processing", this );
	toolbar->addAction( action );
	addToolBar( toolbar );



	addViewerActionSetToWidget( *menuViewer, actions );
	LOG( "Viewer control actions created and added to GUI" );

	mInfoLabel = new QLabel();
	statusbar->addWidget( mInfoLabel );

	//************* VIEWER FACTORY *****************
	M4D::GUI::Viewer::GeneralViewerFactory::Ptr factory = M4D::GUI::Viewer::GeneralViewerFactory::Ptr( new M4D::GUI::Viewer::GeneralViewerFactory );
	factory->setViewerController( mViewerController );
	factory->setRenderingExtension( mRenderingExtension );
	mViewerDesktop->setViewerFactory( factory );
	LOG( "Viewer factory initialized" );

	mViewerDesktop->setLayoutOrganization( 2, 1 );
	QObject::connect( mViewerDesktop, SIGNAL( updateInfo( const QString & ) ), this, SLOT( updateInfoInStatusBar( const QString & ) ), Qt::QueuedConnection );

	QObject::connect( this, SIGNAL( callInitAfterLoopStart() ), this, SLOT( initAfterLoopStart() ), Qt::QueuedConnection );
	emit callInitAfterLoopStart();

	// HH: OIS support
#ifdef OIS_ENABLED
	mJoyInput.startup((size_t)this->winId());
	if(mJoyInput.getNrJoysticks() > 0) {
		mJoyTimer.setInterval( 50 );
		QObject::connect( &mJoyTimer, SIGNAL( timeout() ), this, SLOT( updateJoyControl() ) );
		mJoyTimer.start();
	}
	LOG( "OIS initialized" );
#endif


	QObject::connect( M4D::ApplicationManager::getInstance(), SIGNAL( selectedViewerSettingsChanged() ), this, SLOT( selectedViewerSettingsChanged() ) );

	QObject::connect( M4D::ApplicationManager::getInstance(), SIGNAL( viewerSelectionChanged() ), this, SLOT( changedViewerSelection() ) );

	QObject::connect(mTFPaletteWidget.get(), &tfw::PaletteWidget::transferFunctionAdded, this, &ViewerWindow::transferFunctionAdded, Qt::QueuedConnection);
	QObject::connect(mTFPaletteWidget.get(), &tfw::PaletteWidget::changedTransferFunctionSelection, this, &ViewerWindow::changedTransferFunctionSelection, Qt::QueuedConnection);
	QObject::connect(mTFPaletteWidget.get(), &tfw::PaletteWidget::transferFunctionModified, this, &ViewerWindow::transferFunctionModified, Qt::QueuedConnection);

	LOG( "Basic signals/slots connected" );

	mOpenDialog = new QFileDialog( this, tr("Open Image") );
	mOpenDialog->setFileMode(QFileDialog::ExistingFile);
	//mOpenDialog->setOption(QFileDialog::DontUseNativeDialog, false);

	QObject::connect(&mDatasetManager, &DatasetManager::registeredNewDataset, this, &ViewerWindow::dataLoaded, Qt::QueuedConnection);
}

void
ViewerWindow::denoiseImage()
{
	LOG( "denoiseImage()" );
	M4D::ImageRecord::Ptr rec = M4D::DatasetManager::getInstance()->getCurrentImageInfo();
	M4D::Imaging::AImage::Ptr image = rec->image;
	if( !image ) {
		return;
	}
	#ifdef USE_CUDA
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			IMAGE_TYPE::Ptr outputImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< TTYPE, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );

			median3D( typedImage->GetRegion(), outputImage->GetRegion(), 2 );
			M4D::DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( outputImage );
			rec->image = outputImage;
		);
	#endif
		//TODO - CPU version
}

void
ViewerWindow::initAfterLoopStart()
{
	//toggleInteractiveTransferFunction( true );
}


ViewerWindow::~ViewerWindow()
{
#ifdef OIS_ENABLED
	mJoyInput.destroy();
#endif

}

void
ViewerWindow::updateInfoInStatusBar( const QString &aInfo )
{
	statusbar->showMessage( aInfo ) ;
}

M4D::GUI::Viewer::GeneralViewer *
ViewerWindow::getSelectedViewer()
{
	return NULL;
}

void
ViewerWindow::addRenderingExtension( M4D::GUI::Viewer::RenderingExtension::Ptr aRenderingExtension )
{
	mRenderingExtension->addRenderingExtension( aRenderingExtension );
}

void
ViewerWindow::setViewerController( M4D::GUI::Viewer::AViewerController::Ptr aViewerController )
{
	mViewerController->setController( aViewerController );
}

void
ViewerWindow::testSlot()
{
	//mViewerDesktop->setLayoutOrganization( 2, 2 );
	LOG( "AAAAA" );
	updateGui();

	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	if(pViewer != NULL) {
		//pViewer->toggleFPS();
		static_cast<M4D::GUI::Viewer::GeneralViewer *>(pViewer)->reloadShaders();
	}
}

void ViewerWindow::testSlot2()
{
/*	auto id1 = mDatasetManager.idFromIndex(0);
	auto id2 = mDatasetManager.idFromIndex(1);

	statusbar->showMessage("Computing combined stats...");
	BOOST_SCOPE_EXIT_ALL(this) {
		statusbar->clearMessage();
	};
	auto stats = mDatasetManager.getCombinedStatistics(id1, id2);

	mTFPaletteWidget->setStatistics(stats);*/
}

void
ViewerWindow::updateGui()
{
	mViewerDesktop->updateAllViewers();
}


struct SetTransferFunctionFtor
{
	SetTransferFunctionFtor( vorgl::TransferFunctionBuffer1D::Ptr aTF ): mTF( aTF )
	{ /*empty*/ }

	void
	operator()( M4D::GUI::Viewer::AGLViewer * aViewer )
	{
		M4D::GUI::Viewer::GeneralViewer * viewer = dynamic_cast< M4D::GUI::Viewer::GeneralViewer * >( aViewer );

		if ( viewer ) {
			//viewer->setTransferFunctionBuffer( mTF );
		}
	}

	vorgl::TransferFunctionBuffer1D::Ptr mTF;
};

void
ViewerWindow::applyTransferFunction()
{
	mViewerDesktop->forEachViewer( SetTransferFunctionFtor( mTransferFunctionEditor->GetTransferFunctionBuffer() ) );
}

void
ViewerWindow::updateTransferFunction()
{
	M4D::Common::TimeStamp timestamp = mTransferFunctionEditor->GetTimeStamp();
	if ( timestamp != mLastTimeStamp ) {
		applyTransferFunction();
		mLastTimeStamp = timestamp;
	}
}

#ifdef OIS_ENABLED
void
ViewerWindow::updateJoyControl()
{
	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	static M4D::Common::Clock myTimer;

	M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
	if(pGenViewer != NULL) {
		mJoyInput.updateJoys();
		float dt = myTimer.SecondsPassed();
		myTimer.Reset();

		float fConstant = dt * 4.0f;
		static float alpha = 0, beta = 0;

		int iOffset = 512;
		int iInputValue = mJoyInput.getSlider(0, 0);
		iInputValue = (iInputValue < 0)? M4D::min(iInputValue + iOffset, int(0)): M4D::max(iInputValue - iOffset, int(0));
		beta = (beta) * (1-fConstant) + fConstant * (-iInputValue * M_PI / 32768);
		iInputValue = mJoyInput.getAxis(0, 0);
		iInputValue = (iInputValue < 0)? M4D::min(iInputValue + iOffset, 0): M4D::max(iInputValue - iOffset, 0);
		alpha = (alpha) * (1-fConstant) + fConstant * (iInputValue * M_PI / 32768);
		pGenViewer->cameraOrbit(Vector2f(alpha*0.05f, beta*0.05f));

		float camZ1 = 0;
		iInputValue = mJoyInput.getAxis(0, 1);
		iInputValue = (iInputValue < 0)? M4D::min(iInputValue + iOffset, 0): M4D::max(iInputValue - iOffset, 0); // right stick Y
		camZ1 = fConstant * iInputValue / 65535.0f;
		float camZ2 = 0;
		iInputValue = mJoyInput.getAxis(0, 4);
		iInputValue = (iInputValue < 0)? M4D::min(iInputValue + iOffset, 0): M4D::max(iInputValue - iOffset, 0); // left stick Y
		camZ2 = fConstant * iInputValue / 65535.0f;
		pGenViewer->cameraDolly(1.0f + camZ1 + camZ2);
	}
}
#endif

void
ViewerWindow::toggleInteractiveTransferFunction( bool aChecked )
{
	if ( aChecked ) {
		D_PRINT( "Transfer function - interactive manipulation enabled" );
		mTransFuncTimer.start();
	} else {
		D_PRINT( "Transfer function - interactive manipulation disabled" );
		mTransFuncTimer.stop();
	}
}


void
ViewerWindow::selectedViewerSettingsChanged()
{
	/*M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
	QString text;
	if(pGenViewer != NULL) {
		if( pGenViewer->getViewType() == M4D::GUI::Viewer::vt2DAlignedSlices ) {
			if( pGenViewer->getColorTransformType() == M4D::GUI::Renderer::ctLUTWindow ) {
				Vector2f win = pGenViewer->getLUTWindow();
				text = QString::number( win[0] ) + "/" + QString::number( win[1] );
			}
		}
	}
	mInfoLabel->setText( text );*/
	//LOG( __FUNCTION__ );
}


void
ViewerWindow::changedViewerSelection()
{
	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
	if(pGenViewer != NULL) {
		vorgl::TransferFunctionBufferInfo info = pGenViewer->getTransferFunctionBufferInfo();
		//mTFEditingSystem->selectTransferFunction( info.id );
		mTFPaletteWidget->selectTransferFunction( info.id );
	}
	//LOG( __FUNCTION__ );
}


struct FillTFBufferVisitor :
	public tfw::UnsupportedThrowConstTransferFunctionVisitor
{
	FillTFBufferVisitor(vorgl::TransferFunctionBufferInfo &aInfo)
		: mInfo(aInfo)
	{}

	void
	visit(const tfw::TransferFunction1D &aTransferFunction) override {
		STUBBED("use some adaptive sample count");
		static const int cSampleCount = 1000;
		float step = (aTransferFunction.range().second - aTransferFunction.range().first) / cSampleCount;
		auto tfBuffer = vorgl::TransferFunctionBuffer1D(cSampleCount);
		tfBuffer.setMappedInterval(vorgl::TransferFunctionBuffer1D::MappedInterval(aTransferFunction.range().first, aTransferFunction.range().second));
		for (size_t i = 0; i < tfBuffer.size(); ++i) {
			vorgl::RGBAf color;
			double value = aTransferFunction.range().first + i * step;
			color.r = aTransferFunction.getIntensity(value, 0);
			color.g = aTransferFunction.getIntensity(value, 1);
			color.b = aTransferFunction.getIntensity(value, 2);
			color.a = aTransferFunction.getIntensity(value, 3);
			tfBuffer[i] = color;
		}

		auto tfIntegralBuffer = vorgl::TransferFunctionBuffer1D(cSampleCount);
		tfIntegralBuffer.setMappedInterval(tfBuffer.mappedInterval());
		vorgl::RGBAf lastColor = tfBuffer.front();
		for (size_t i = 1; i < tfBuffer.size(); ++i) {
			vorgl::RGBAf color = tfBuffer[i];
			vorgl::RGBAf tmpColor = (lastColor + color) * 0.5f;
			float alpha = /*1.0f;/*/tmpColor.a;
			tfIntegralBuffer[i] = tfIntegralBuffer[i - 1] + vorgl::TransferFunctionBuffer1D::value_type(
				tmpColor.r * alpha,
				tmpColor.g * alpha,
				tmpColor.b * alpha,
				tmpColor.a);
			lastColor = color;
		}


		OpenGLManager::getInstance()->doGL([this, &tfBuffer, &tfIntegralBuffer]() {
				vorgl::TransferFunctionBuffer1DInfo info;
				info.tfGLBuffer = vorgl::createGLTransferFunctionBuffer1D(tfBuffer);
				info.tfGLIntegralBuffer = vorgl::createGLTransferFunctionBuffer1D(tfIntegralBuffer);
				mInfo.bufferInfo = info;
			});
	}

	void
	visit(const tfw::TransferFunction2D &aTransferFunction) override {
		STUBBED("Handle these constants for 2D transfer function");
		static const int cXSampleCount = 1000;
		static const int cYSampleCount = 200;
		std::vector<vorgl::RGBAf> buffer(cXSampleCount * cYSampleCount);
		tfw::TransferFunction2D::RangePoint from, to;
		std::tie(from, to) = aTransferFunction.range();
		std::array<float, 2> step {
			(to[0] - from[0]) / cXSampleCount,
			(to[1] - from[1]) / cYSampleCount };
		for (int j = 0; j < cYSampleCount; ++j) {
			for (int i = 0; i < cXSampleCount; ++i) {
				buffer[j*cXSampleCount + i] = aTransferFunction.getColor(i * step[0] + from[0], j * step[1] + from[1]).data();
			}
		}
		OpenGLManager::getInstance()->doGL([this, &buffer, from, to] () {
				vorgl::TransferFunctionBuffer2DInfo info;
				info.tfGLBuffer = vorgl::createGLTransferFunctionBuffer2D(
							buffer.data(),
							cXSampleCount,
							cYSampleCount,
							glm::fvec2(std::get<0>(from), std::get<1>(from)),
							glm::fvec2(std::get<0>(to), std::get<1>(to)));
				mInfo.bufferInfo = info;
			});
	}

	vorgl::TransferFunctionBufferInfo &mInfo;
};



bool
fillTransferFunctionInfo(const tfw::ATransferFunction &function, vorgl::TransferFunctionBufferInfo &info )
{
	try {
		FillTFBufferVisitor fillBufferVisitor(info);
		function.accept(fillBufferVisitor);

	} catch (tfw::EUnsupportedTransferFunctionType &) {
		D_PRINT( "TF buffer not created" );
		return false;
	}
	return true;
}

void
ViewerWindow::transferFunctionAdded(int idx)
{
	vorgl::TransferFunctionBufferInfo info;
	info.id = idx;

	//if (fillTransferFunctionInfo(mTFEditingSystem->getTransferFunction(idx), info ) ) {
	if (fillTransferFunctionInfo(mTFPaletteWidget->getTransferFunction(idx), info)) {
		TransferFunctionBufferUsageRecord record;
		record.info = info;
		mTFUsageMap[idx] = record;
	}
}

void
ViewerWindow::changedTransferFunctionSelection()
{
	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
	if(pGenViewer) {
		int idx = mTFPaletteWidget->getSelectedTransferFunctionIndex();
		vorgl::TransferFunctionBufferInfo oldInfo = pGenViewer->getTransferFunctionBufferInfo();
		if (idx == oldInfo.id) {
			return; //No change
		}
		auto it = mTFUsageMap.find( oldInfo.id );
		if (it != end(mTFUsageMap)) {
			it->second.viewers.remove(pGenViewer);
		}
		it = mTFUsageMap.find(idx);
		if(it == end(mTFUsageMap)) {
			transferFunctionAdded(idx);
		}
		it = mTFUsageMap.find( idx );
		if ( it != mTFUsageMap.end() ) {
			pGenViewer->setTransferFunctionBufferInfo(it->second.info);
			it->second.viewers.push_back(pGenViewer);
		} else {
			LOG( "Function not found" );
		}
	}
	//LOG( __FUNCTION__ );
}

void
ViewerWindow::transferFunctionModified( int idx )
{
	TransferBufferUsageMap::iterator it = mTFUsageMap.find( idx );
	if ( it != end(mTFUsageMap)) {
		TransferFunctionBufferUsageRecord &rec = it->second;
		fillTransferFunctionInfo(mTFPaletteWidget->getTransferFunction(idx), rec.info);
		for (auto viewer : rec.viewers) {
			viewer->setTransferFunctionBufferInfo(rec.info);
		}
	} else {
		D_PRINT( "Modified function not found" );
	}
}


void
ViewerWindow::showSettingsDialog()
{
	SettingsDialog *dialog = new SettingsDialog( this );
	dialog->showDialog( M4D::ApplicationManager::getInstance()->settings() );
	delete dialog;
}

void
ViewerWindow::openFile()
{
	auto id = mDatasetManager.loadFromFile();
	if (!id) {
		return;
	}
	//dataLoaded(id);
}

void ViewerWindow::closeAllFiles()
{
	//TODO - add confirmation dialog
	mViewerDesktop->forEachViewer(
		[](M4D::GUI::Viewer::AGLViewer * aViewer) {
			M4D::GUI::Viewer::GeneralViewer * viewer = dynamic_cast< M4D::GUI::Viewer::GeneralViewer * >( aViewer );
			if (viewer) {
				viewer->setInputData(ViewerInputDataWithId::Ptr());
			}
		});
	mDatasetManager.closeAll();
}

void
ViewerWindow::dataLoaded(DatasetID aId, bool aQuiet)
{
	//M4D::Imaging::AImage::Ptr image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( mDicomObjSet );
	D_PRINT( "Loaded dataset ID = " << aId );
	auto & rec = mDatasetManager.getDatasetRecord(aId);
	M4D::Imaging::AImage::Ptr image = rec.mImage;

	auto inputData = ViewerInputDataWithId::Ptr(new ViewerInputDataWithId(std::static_pointer_cast<M4D::Imaging::AImageDim<3>>(image), aId));

	if (!aQuiet) {
		mViewerDesktop->forEachViewer(
			[inputData](M4D::GUI::Viewer::AGLViewer * aViewer) {
				M4D::GUI::Viewer::GeneralViewer * viewer = dynamic_cast< M4D::GUI::Viewer::GeneralViewer * >( aViewer );
				if (viewer) {
					viewer->setInputData(inputData);
				}
			});
	}
	//M4D::DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( image );

	computeHistogram(aId/* image */);
}

void ViewerWindow::processModule(AModule &aModule)
{
	//TODO - do proper type handling
	auto & module = static_cast<ViewerModule &>(aModule);
	module.setDatasetManager(mDatasetManager);
}

void
ViewerWindow::computeHistogram(DatasetID aId/* M4D::Imaging::AImage::Ptr aImage */)
{
	using namespace M4D::Imaging;
	/*if( !aImage ) {
		return;
	}*/
	statusbar->showMessage("Computing histogram...");
	BOOST_SCOPE_EXIT_ALL(this) {
		statusbar->clearMessage();
	};

	//M4D::Common::Clock clock;

	/*Histogram1D<int> histogram1D;
	ScatterPlot2D<int, float> gradientScatterPlot;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( aImage,
		histogram1D = M4D::Imaging::createHistogramForImageRegion2<Histogram1D<int>, IMAGE_TYPE >( IMAGE_TYPE::Cast( *aImage ) );
		//gradientScatterPlot = M4D::Imaging::createGradientScatterPlotForImageRegion<ScatterPlot2D<int, float>, IMAGE_TYPE::SubRegion>(IMAGE_TYPE::Cast( *aImage ).GetRegion());
	);

	auto statistics = std::make_shared<ImageStatistics>();

	statistics->mHistogram = std::move(histogram1D);
	//statistics->mGradientScatterPlot = std::move(gradientScatterPlot);*/
	//mTFPaletteWidget->setStatistics(statistics);

	//mTFPaletteWidget->setStatistics(mDatasetManager.getImageStatistics(aId));

	//LOG( "Histogram computed in " << clock.SecondsPassed() );


	//statusbar->showMessage("Applying transfer function...");
	//applyTransferFunction();

}

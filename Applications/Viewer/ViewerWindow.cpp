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
#include "MedV4D/Common/MathTools.h"
#include <boost/thread.hpp>

#include <iostream>
#include <iterator>
#include <algorithm>



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
	GeneralViewerFactory() :mPrimaryConnection( NULL ), mSecondaryConnection( NULL )
	{}
	typedef std::shared_ptr< GeneralViewerFactory > Ptr;

	AGLViewer *
	createViewer()
	{
		GeneralViewer *viewer = new GeneralViewer();
		if( mRenderingExtension ) {
			viewer->setRenderingExtension( mRenderingExtension );
		}
		if( mViewerController ) {
			viewer->setViewerController( mViewerController );
		}
		if( mPrimaryConnection ) {
			mPrimaryConnection->ConnectConsumer( viewer->InputPort()[0] );
		}
		if( mSecondaryConnection ) {
			mSecondaryConnection->ConnectConsumer( viewer->InputPort()[1] );
		}

		viewer->setLUTWindow(glm::fvec2(1500.0f,100.0f));

		viewer->enableShading( GET_SETTINGS( "gui.viewer.volume_rendering.shading_enabled", bool, true ) );

		viewer->enableJittering( GET_SETTINGS( "gui.viewer.volume_rendering.jittering_enabled", bool, true ) );

		viewer->setJitterStrength( GET_SETTINGS( "gui.viewer.volume_rendering.jitter_strength", double, 1.0 ) );

		viewer->enableIntegratedTransferFunction( GET_SETTINGS( "gui.viewer.volume_rendering.integrated_transfer_function_enabled", bool, true ) );

		viewer->setRenderingQuality( GET_SETTINGS( "gui.viewer.volume_rendering.rendering_quality", int, qmNormal ) );

		viewer->enableBoundingBox( GET_SETTINGS( "gui.viewer.volume_rendering.bounding_box_enabled", bool, true ) );

		Vector4d color = GET_SETTINGS( "gui.viewer.background_color", Vector4d, Vector4d( 0.0, 0.0, 0.0, 1.0 ) );
		viewer->setBackgroundColor( QColor::fromRgbF( color[0], color[1], color[2], color[3] ) );


		//viewer->setBackgroundColor( QColor( 50,50,100 ) );
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
	void
	setPrimaryInputConnection( M4D::Imaging::ConnectionInterface &mProdconn )
	{
		mPrimaryConnection = &mProdconn;
	}
	void
	setSecondaryInputConnection( M4D::Imaging::ConnectionInterface &mProdconn )
	{
		mSecondaryConnection = &mProdconn;
	}
protected:
	RenderingExtension::Ptr mRenderingExtension;
	AViewerController::Ptr	mViewerController;
	M4D::Imaging::ConnectionInterface *mPrimaryConnection;
	M4D::Imaging::ConnectionInterface *mSecondaryConnection;
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


bool fillBufferFromTF(M4D::GUI::TransferFunctionInterface::Const function, vorgl::TransferFunctionBuffer1D::Ptr& buffer){

	if(!function) return false;

	M4D::GUI::TF::Size domain = function.getDomain(TF_DIMENSION_1);
	if(!buffer || buffer->size() != domain)
	{
		buffer = vorgl::TransferFunctionBuffer1D::Ptr(new vorgl::TransferFunctionBuffer1D(domain, vorgl::TransferFunctionBuffer1D::MappedInterval(0.0f, (float)domain)));
	}

	M4D::GUI::TF::Coordinates coords(1);
	M4D::GUI::TF::Color color;
	for(M4D::GUI::TF::Size i = 0; i < domain; ++i)
	{
		coords[0] = i;
		color = function.getRGBfColor(coords);

		(*buffer)[i] = vorgl::TransferFunctionBuffer1D::value_type(
			color.component1,
			color.component2,
			color.component3,
			color.alpha);
	}
	return true;
}

bool fillIntegralBufferFromTF(M4D::GUI::TransferFunctionInterface::Const function, vorgl::TransferFunctionBuffer1D::Ptr& buffer){

	if(!function) return false;

	M4D::GUI::TF::Size domain = function.getDomain(TF_DIMENSION_1);
	if(!buffer || buffer->size() != domain)
	{
		buffer = vorgl::TransferFunctionBuffer1D::Ptr(new vorgl::TransferFunctionBuffer1D(domain, vorgl::TransferFunctionBuffer1D::MappedInterval(0.0f, (float)domain)));
	}

	M4D::GUI::TF::Coordinates coords(1);
	M4D::GUI::TF::Color lastColor = function.getRGBfColor(coords);
	M4D::GUI::TF::Color color;
	for(M4D::GUI::TF::Size i = 1; i < domain; ++i)
	{
		coords[0] = i;
		color = function.getRGBfColor(coords);

		M4D::GUI::TF::Color tmpColor = (lastColor + color)*0.5f;
		float alpha = 1.0f;//tmpColor.alpha;
		(*buffer)[i] = (*buffer)[i-1] + vorgl::TransferFunctionBuffer1D::value_type(
			tmpColor.component1 * alpha,
			tmpColor.component2 * alpha,
			tmpColor.component3 * alpha,
			tmpColor.alpha);;
		lastColor = color;
	}
	return true;
}

void
loadAllSavedTFEditorsIntoPalette( M4D::GUI::Palette &palette, boost::filesystem::path dirName )
{
	if (!boost::filesystem::exists(dirName)) {
		LOG( "Directory \'" << dirName << "\' doesn't exist!" );
		return;
	}
	if (!boost::filesystem::is_directory(dirName) ){
		LOG( "\'" << dirName << "\' is not a directory!" );
		return;
	}

	boost::filesystem::directory_iterator dirIt(dirName);
	boost::filesystem::directory_iterator end;
	for ( ;dirIt != end; ++dirIt ) {
		LOG( "Found TFE file :" << *dirIt );
		boost::filesystem::path p = dirIt->path();
		palette.loadFromFile( QString( p.string().data() ), false );
	}
}


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
	mTFPalette->add(std::make_shared<tfw::TransferFunction1D>(0.0, 2000.0));
	mTFPalette->add(std::make_shared<tfw::TransferFunction1D>(0.0, 2000.0));
	mTFPaletteWidget->setPalette(mTFPalette);
	createDockWidget( tr("Transfer Function Palette New"), Qt::RightDockWidgetArea, mTFPaletteWidget.get(), true );

	mTFPaletteWidget->setWrapEditorCallback(
			[this](tfw::ATransferFunctionEditor *aEditor) -> QWidget * {
				return createDockWidget(aEditor->tfName(), Qt::RightDockWidgetArea, aEditor, true );
			});
//*****************

	/*mTransferFunctionEditor = new M4D::GUI::TransferFunction1DEditor;
	createDockWidget( tr("Transfer Function" ), Qt::RightDockWidgetArea, mTransferFunctionEditor );

	mTransferFunctionEditor->SetValueInterval( 0.0f, 3000.0f );
	mTransferFunctionEditor->SetMappedValueInterval( 0.0f, 1.0f );
	mTransferFunctionEditor->SetBorderWidth( 10 );

	mTransFuncTimer.setInterval( 500 );
	QObject::connect( &mTransFuncTimer, SIGNAL( timeout() ), this, SLOT( updateTransferFunction() ) );*/
//***********************************************************
#ifdef USE_PYTHON
	M4D::GUI::TerminalWidget *mTerminal = new M4D::GUI::PythonTerminal;
	createDockWidget( tr("Python Terminal" ), Qt::BottomDockWidgetArea, mTerminal, false );
	LOG( "Python terminal initialized" );
#endif //USE_PYTHON

	ViewerControls *mViewerControls = new ViewerControls;
	QObject::connect(
			ApplicationManager::getInstance(),
			&ApplicationManager::viewerSelectionChanged,
			[mViewerControls] () {
				auto viewer = ViewerManager::getInstance()->getSelectedViewer();
				auto genViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (viewer);
				mViewerControls->setViewer(genViewer);
			});
	QObject::connect( ApplicationManager::getInstance(), SIGNAL( selectedViewerSettingsChanged() ), mViewerControls, SLOT( updateControls() ) );
	createDockWidget( tr("Viewer Controls" ), Qt::RightDockWidgetArea, mViewerControls );
	LOG( "Viewer controls GUI initialized" );

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
	factory->setPrimaryInputConnection( DatasetManager::getInstance()->primaryImageInputConnection() );
	factory->setSecondaryInputConnection( DatasetManager::getInstance()->secondaryImageInputConnection() );
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


	QObject::connect( ApplicationManager::getInstance(), SIGNAL( selectedViewerSettingsChanged() ), this, SLOT( selectedViewerSettingsChanged() ) );

	QObject::connect( ApplicationManager::getInstance(), SIGNAL( viewerSelectionChanged() ), this, SLOT( changedViewerSelection() ) );

	/*QObject::connect( mTFEditingSystem.get(), SIGNAL(transferFunctionAdded( int ) ), this, SLOT( transferFunctionAdded( int ) ), Qt::QueuedConnection );
	QObject::connect( mTFEditingSystem.get(), SIGNAL(changedTransferFunctionSelection( int ) ), this, SLOT( changedTransferFunctionSelection() ), Qt::QueuedConnection );
	QObject::connect( mTFEditingSystem.get(), SIGNAL(transferFunctionModified( int )), this, SLOT( transferFunctionModified( int ) ), Qt::QueuedConnection );
*/
	QObject::connect(mTFPaletteWidget.get(), &tfw::PaletteWidget::transferFunctionAdded, this, &ViewerWindow::transferFunctionAdded, Qt::QueuedConnection);
	QObject::connect(mTFPaletteWidget.get(), &tfw::PaletteWidget::changedTransferFunctionSelection, this, &ViewerWindow::changedTransferFunctionSelection, Qt::QueuedConnection);
	QObject::connect(mTFPaletteWidget.get(), &tfw::PaletteWidget::transferFunctionModified, this, &ViewerWindow::transferFunctionModified, Qt::QueuedConnection);

	LOG( "Basic signals/slots connected" );

	mOpenDialog = new QFileDialog( this, tr("Open Image") );
	mOpenDialog->setFileMode(QFileDialog::ExistingFile);
	//mOpenDialog->setOption(QFileDialog::DontUseNativeDialog, false);

}

void
ViewerWindow::denoiseImage()
{
	LOG( "denoiseImage()" );
	ImageRecord::Ptr rec = DatasetManager::getInstance()->getCurrentImageInfo();
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
			DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( outputImage );
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
	LOG( "AAAAA" );
	updateGui();

	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	if(pViewer != NULL) {
		//pViewer->toggleFPS();
		static_cast<M4D::GUI::Viewer::GeneralViewer *>(pViewer)->reloadShaders();
	}
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
			float alpha = 1.0f;//tmpColor.alpha;
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
		vorgl::TransferFunctionBuffer2DInfo info;
		mInfo.bufferInfo = info;
		//vorgl::createGLTransferFunctionBuffer2D(*mInfo.tfIntegralBuffer);
	}

	vorgl::TransferFunctionBufferInfo &mInfo;
};



bool
fillTransferFunctionInfo(/*M4D::GUI::TransferFunctionInterface::Const*/const tfw::ATransferFunction &function, vorgl::TransferFunctionBufferInfo &info )
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

		//fillTransferFunctionInfo( mTFEditingSystem->getTransferFunction(idx), rec.info );
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
	dialog->showDialog( ApplicationManager::getInstance()->settings() );
	delete dialog;
}

void
ViewerWindow::openFile()
{
	try {
	QStringList fileNames;
	QString fileName;
	if ( mOpenDialog->exec() ) {
		fileNames = mOpenDialog->selectedFiles();
		if (!fileNames.isEmpty()) {
			fileName = fileNames[0];
		}
	}
	if ( !fileName.isEmpty() ) {
		if( !mProgressDialog ) {
			mProgressDialog = ProgressInfoDialog::Ptr( new ProgressInfoDialog( this ) );
			QObject::connect( mProgressDialog.get(), SIGNAL( finishedSignal() ), this, SLOT( dataLoaded() ), Qt::QueuedConnection );
		}

		mDatasetId = DatasetManager::getInstance()->openFileNonBlocking( std::string( fileName.toLocal8Bit().data() ), mProgressDialog );

		mProgressDialog->show();
	}
	} catch ( std::exception &e ) {
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	}
	catch (...) {
		QMessageBox::critical ( NULL, "Exception", "Problem with file loading" );
	}
}

void
ViewerWindow::openFile( const QString &aPath )
{
	ASSERT( false );
	std::string path = std::string( aPath.toLocal8Bit().data() );
	M4D::Imaging::AImage::Ptr image = M4D::Imaging::ImageFactory::LoadDumpedImage( path );
	DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( image );

	statusbar->showMessage("Computing histogram...");
	M4D::Common::Clock clock;

	M4D::Imaging::Histogram1D<int> histogram2;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		histogram2 = M4D::Imaging::createHistogramForImageRegion2<M4D::Imaging::Histogram1D<int>, IMAGE_TYPE>(IMAGE_TYPE::Cast(*image));
	);
	LOG( "Histogram computed in " << clock.SecondsPassed() );

	statusbar->clearMessage();
}

void
ViewerWindow::openDicom( const QString &aPath )
{
	ASSERT( false );
	LOG( "Opening Dicom" );
	std::string path = std::string( aPath.toLocal8Bit().data() );

	QFileInfo pathInfo( aPath );

	if( !mDicomObjSet ) {
		mDicomObjSet = M4D::Dicom::DicomObjSetPtr( new M4D::Dicom::DicomObjSet() );
	}

	if( !mProgressDialog ) {
		mProgressDialog = ProgressInfoDialog::Ptr( new ProgressInfoDialog( this ) );
		QObject::connect( mProgressDialog.get(), SIGNAL( finishedSignal() ), this, SLOT( dataLoaded() ), Qt::QueuedConnection );
	}
	/*boost::thread th = boost::thread(
			&M4D::Dicom::DcmProvider::LoadSerieThatFileBelongsTo,
			std::string( pathInfo.absoluteFilePath().toLocal8Bit().data() ),
			std::string( pathInfo.absolutePath().toLocal8Bit().data() ),
			boost::ref( *mDicomObjSet ),
			mProgressDialog
			);
	th.detach();*/
	mProgressDialog->show();
	//th.join();
}

void
ViewerWindow::dataLoaded()
{
	//M4D::Imaging::AImage::Ptr image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( mDicomObjSet );
	D_PRINT( "Loaded dataset ID = " << mDatasetId );
	ADatasetRecord::Ptr rec = DatasetManager::getInstance()->getDatasetInfo( mDatasetId );
	if ( !rec ) {
		D_PRINT( "Loaded dataset record not available" );
		return;
	}
	ImageRecord * iRec = dynamic_cast< ImageRecord * >( rec.get() );
	if ( !iRec ) {
		D_PRINT( "Loaded dataset isn't image" );
	}
	M4D::Imaging::AImage::Ptr image = iRec->image;

	DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( image );

	computeHistogram( image );
}

class ImageStatistics : public tfw::AStatistics
{
public:
	bool
	hasHistogram() const override
	{
		return true;
	}

	std::pair<float, float>
	getHistogramRange() const override
	{
		auto range = mHistogram.getRange();
		return std::make_pair(float(range.first), float(range.second));
	}

	virtual std::vector<QPointF>
	getHistogramSamples() const override
	{
		auto extremes = mHistogram.minmax();
		std::vector<QPointF> points;
		points.reserve(mHistogram.resolution()[0]);
		float step = float(mHistogram.getRange().second - mHistogram.getRange().first) / mHistogram.resolution()[0];
		float x = 0.0f;//TODO
		for (auto value : mHistogram) {
			points.emplace_back(x, float(value) / extremes.second);
			x += step;
		}
		return points;
	}

	bool
	hasScatterPlot() const override
	{
		return true;
	}

	std::pair<QRectF, tfw::ScatterPlotData>
	getScatterPlot() const override
	{
		auto range = mGradientScatterPlot.getRange();
		QRectF region(
			range.first.first,
			range.first.second,
			range.second.first - range.first.first,
			range.second.second - range.first.second);

		auto minmax = mGradientScatterPlot.minmax();
		auto resolution = mGradientScatterPlot.resolution();

		tfw::ScatterPlotData data;
		data.size[0] = resolution[0];
		data.size[1] = resolution[1];
		data.buffer.resize(resolution[0] * resolution[1]);

		for (int j = 0; j < resolution[1]; ++j) {
			for (int i = 0; i < resolution[0]; ++i) {
				data.buffer[i + j*resolution[0]] = double(mGradientScatterPlot.data()[i + j*resolution[0]]) / minmax.second;
			}
		}
		return std::make_pair(region, std::move(data));
	}

	M4D::Imaging::Histogram1D<int> mHistogram;
	M4D::Imaging::ScatterPlot2D<int, float> mGradientScatterPlot;
};

void
ViewerWindow::computeHistogram( M4D::Imaging::AImage::Ptr aImage )
{
	using namespace M4D::Imaging;
	if( !aImage ) {
		return;
	}
	statusbar->showMessage("Computing histogram...");
	M4D::Common::Clock clock;

	Histogram1D<int> histogram1D;
	ScatterPlot2D<int, float> gradientScatterPlot;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( aImage,
		histogram1D = M4D::Imaging::createHistogramForImageRegion2<Histogram1D<int>, IMAGE_TYPE >( IMAGE_TYPE::Cast( *aImage ) );
		gradientScatterPlot = M4D::Imaging::createGradientScatterPlotForImageRegion<ScatterPlot2D<int, float>, IMAGE_TYPE::SubRegion>(IMAGE_TYPE::Cast( *aImage ).GetRegion());
	);

	auto statistics = std::make_shared<ImageStatistics>();

	statistics->mHistogram = std::move(histogram1D);
	statistics->mGradientScatterPlot = std::move(gradientScatterPlot);
	mTFPaletteWidget->setStatistics(statistics);

	LOG( "Histogram computed in " << clock.SecondsPassed() );


	//statusbar->showMessage("Applying transfer function...");
	//applyTransferFunction();

	statusbar->clearMessage();
}

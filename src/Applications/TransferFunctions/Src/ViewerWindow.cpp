#include "ViewerWindow.hpp"


ViewerWindow::ViewerWindow():
	fileLoaded_(false){

	setupUi( this );
	showMaximized();

	#ifdef WIN32
		//Reposition console window
		QRect myRegion=frameGeometry();
		QPoint putAt=myRegion.topRight();
		SetWindowPos(GetConsoleWindow(),winId(),putAt.x()+1,putAt.y(),0,0,SWP_NOSIZE);
	#endif

	mProdconn.ConnectConsumer( mViewer->InputPort()[0] );

	mViewer->SetLUTWindow( Vector2f( 500.0f,1000.0f ) );
	
	//---TF Editor---

	mTransferFunctionEditor = new M4D::GUI::TFPalette(this);
	//mTransferFunctionEditor->setupDefault();	

	QDockWidget* dockWidget = new QDockWidget("Transfer Function Palette", this);
	
	dockWidget->setWidget( mTransferFunctionEditor );
	dockWidget->setFeatures(QDockWidget::AllDockWidgetFeatures);
	dockWidget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
	
	addDockWidget(Qt::LeftDockWidgetArea, dockWidget);

	//---Timer---

	mTransFuncTimer.setInterval( 500 );
	QObject::connect( &mTransFuncTimer, SIGNAL( timeout() ), this, SLOT( updateTransferFunction() ) );

	//---Viewer Switch---

	QActionGroup *viewerTypeSwitch = new QActionGroup( this );
	QSignalMapper *viewerTypeSwitchSignalMapper = new QSignalMapper( this );
	
	viewerTypeSwitch->setExclusive( true );	
	viewerTypeSwitch->addAction( action2D );
	viewerTypeSwitch->addAction( action3D );
	
	viewerTypeSwitchSignalMapper->setMapping( action2D, M4D::GUI::rt2DAlignedSlices );
	viewerTypeSwitchSignalMapper->setMapping( action3D, M4D::GUI::rt3D );
	
	QObject::connect( action2D, SIGNAL( triggered() ), viewerTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( action3D, SIGNAL( triggered() ), viewerTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( viewerTypeSwitchSignalMapper, SIGNAL( mapped ( int ) ), this, SLOT( changeViewerType( int ) ) );

	//---Color mapping Swith---

	QActionGroup *colorMapTypeSwitch = new QActionGroup( this );
	QSignalMapper *colorMapTypeSwitchSignalMapper = new QSignalMapper( this );
	
	colorMapTypeSwitch->setExclusive( true );	
	colorMapTypeSwitch->addAction( actionUse_WLWindow );
	colorMapTypeSwitch->addAction( actionUse_Simple_Colormap );
	colorMapTypeSwitch->addAction( actionUse_MIP );
	colorMapTypeSwitch->addAction( actionUse_Transfer_Function );
	
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_WLWindow, M4D::GUI::ctLUTWindow );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_Simple_Colormap, M4D::GUI::ctSimpleColorMap );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_MIP, M4D::GUI::ctMaxIntensityProjection );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_Transfer_Function, M4D::GUI::ctTransferFunction1D );
	
	QObject::connect( actionUse_WLWindow, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( actionUse_Simple_Colormap, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( actionUse_MIP, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( actionUse_Transfer_Function, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( colorMapTypeSwitchSignalMapper, SIGNAL( mapped ( int ) ), this, SLOT( changeColorMapType( int ) ) );

	//---Update---

	updateToolbars();

	//---Mouse info---

	QLabel *infoLabel = new QLabel();
	statusbar->addWidget( infoLabel );

	mViewer->setMouseTracking ( true );
	QObject::connect( mViewer, SIGNAL( MouseInfoUpdate( const QString & ) ), infoLabel, SLOT( setText( const QString & ) ) );
		
	//---Default buffer---

	buffer_ = Buffer1DPtr(new Buffer1D(4095, Interval(0.0f, 4095.0f)));	

	for(Buffer1D::Iterator it = buffer_->Begin(); it != buffer_->End(); ++it)
	{		
		*it = Buffer1D::ValueType(0,0,0,0);
	}
	mViewer->SetTransferFunctionBuffer(buffer_);
}

ViewerWindow::~ViewerWindow(){}

void ViewerWindow::changeViewerType( int aRendererType )
{
	D_PRINT( "Change viewer type" );
	if ( aRendererType == M4D::GUI::rt3D 
		&& ( mViewer->GetColorTransformType() == M4D::GUI::ctLUTWindow 
		|| mViewer->GetColorTransformType() == M4D::GUI::ctSimpleColorMap ) ) 
	{
		changeColorMapType( M4D::GUI::ctTransferFunction1D );
	}

	if ( aRendererType == M4D::GUI::rt2DAlignedSlices 
		&& mViewer->GetColorTransformType() == M4D::GUI::ctMaxIntensityProjection ) 
	{
		changeColorMapType( M4D::GUI::ctLUTWindow );
	}

	mViewer->SetRendererType( aRendererType );
	updateToolbars();
}

void ViewerWindow::changeColorMapType( int aColorMap ){

	D_PRINT( "Change color map type" );

	mViewer->SetColorTransformType( aColorMap );
	updateToolbars();
}

void ViewerWindow::applyTransferFunction(){

	if(!fileLoaded_) return;

	if(buffer_ && (buffer_->Size() != mTransferFunctionEditor->getDomain()))
	{
		buffer_ = Buffer1DPtr(new Buffer1D(mTransferFunctionEditor->getDomain(),
			Interval(0.0f, (float)mTransferFunctionEditor->getDomain())));
	}
	
	bool tfUsed = mTransferFunctionEditor->applyTransferFunction<Buffer1D::iterator>( buffer_->Begin(), buffer_->End());
	
	if(tfUsed) mViewer->SetTransferFunctionBuffer(buffer_);
}

void ViewerWindow::updateTransferFunction(){

	const M4D::Common::TimeStamp timestamp = mTransferFunctionEditor->getTimeStamp();

	if (timestamp != mLastTimeStamp )
	{
		applyTransferFunction();
		mLastTimeStamp = timestamp;
	}
}

void ViewerWindow::toggleInteractiveTransferFunction( bool aChecked )
{
	if ( aChecked )
	{
		D_PRINT( "Transfer function - interactive manipulation enabled" );
		mTransFuncTimer.start();
	}
	else
	{
		D_PRINT( "Transfer function - interactive manipulation disabled" );
		mTransFuncTimer.stop();
	}
}

void
ViewerWindow::updateToolbars()
{
	int rendererType = mViewer->GetRendererType();

	if(rendererType != 0 && rendererType != 2) rendererType = 0;

	switch ( rendererType )
	{
		case M4D::GUI::rt2DAlignedSlices:
		{
			action2D->setChecked( true );
			actionUse_WLWindow->setEnabled( true );
			actionUse_Simple_Colormap->setEnabled( true );
			actionUse_MIP->setEnabled( false );
			break;
		}
		case M4D::GUI::rt3DGeneralSlices:
		{
			ASSERT( false );
			break;
		}
		case M4D::GUI::rt3D:
		{
			action3D->setChecked( true );
			actionUse_WLWindow->setEnabled( false );
			actionUse_Simple_Colormap->setEnabled( false );
			actionUse_MIP->setEnabled( true );
			break;
		}
		default:
		{
			ASSERT( false );
		}
	}

	int colorTransformType = mViewer->GetColorTransformType();

	switch ( colorTransformType )
	{
		case M4D::GUI::ctLUTWindow:
		{
			actionUse_WLWindow->setChecked( true );
			break;
		}
		case M4D::GUI::ctSimpleColorMap:
		{
			actionUse_Simple_Colormap->setChecked( true );
			break;
		}
		case M4D::GUI::ctTransferFunction1D:
		{
			actionUse_Transfer_Function->setChecked( true );
			break;
		}
		case M4D::GUI::ctMaxIntensityProjection:
		{
			actionUse_MIP->setChecked( true );
			break;
		}
		default:
		{
			ASSERT( false );
		}
	}
}

void ViewerWindow::openFile()
{	
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Image"),
		QDir::currentPath(),
		QObject::tr("Dump Files (*.dump)"));

	if ( !fileName.isEmpty() ) {
		openFile( fileName );
	}
}

void ViewerWindow::openFile( const QString &aPath )
{
	std::string path = std::string( aPath.toLocal8Bit().data() );
	M4D::Imaging::AImage::Ptr image = M4D::Imaging::ImageFactory::LoadDumpedImage( path );
	mProdconn.PutDataset( image );
	fileLoaded_ = true;
	
	M4D::Common::Clock clock;

	Histogram::Ptr histogram;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		histogram = M4D::Imaging::CreateHistogramForImageRegion<Histogram, IMAGE_TYPE >( IMAGE_TYPE::Cast( *image ) );
	);
	LOG( "Histogram computed in " << clock.SecondsPassed() );
	
	TFHistogram::Ptr tfHistogram(new TFHistogram);
	for(Histogram::iterator it = histogram->Begin(); it != histogram->End(); ++it)
	{
		tfHistogram->add(*it);
	}
	bool histogramSet = mTransferFunctionEditor->setHistogram(tfHistogram);	
	
	assert(histogramSet);

	mViewer->ZoomFit();
	updateTransferFunction();
	mTransFuncTimer.start();
}
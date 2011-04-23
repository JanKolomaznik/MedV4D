#include "ViewerWindow.hpp"


ViewerWindow::ViewerWindow():
	fileLoaded_(false){

	setupUi( this );

	#ifdef WIN32
		//Reposition console window
		QRect myRegion=frameGeometry();
		QPoint putAt=myRegion.topRight();
		SetWindowPos(GetConsoleWindow(),winId(),putAt.x()+1,putAt.y(),0,0,SWP_NOSIZE);
	#endif

	mProdconn.ConnectConsumer( mViewer->InputPort()[0] );

	mViewer->SetLUTWindow( Vector2f( 500.0f,1000.0f ) );
	
	//---TF Editor---

	editingSystem_ = M4D::GUI::TFPalette::Ptr(new M4D::GUI::TFPalette(this));
	editingSystem_->setupDefault();	

	bool previewUpdateConnected = QObject::connect(&(*editingSystem_), SIGNAL(UpdatePreview(M4D::GUI::TF::Size)),
		this, SLOT(updatePreview(M4D::GUI::TF::Size)));
	tfAssert(previewUpdateConnected);

	QDockWidget* dockWidget = new QDockWidget("Transfer Function Palette", this);
	
	dockWidget->setWidget( &(*editingSystem_) );
	dockWidget->setFeatures(QDockWidget::AllDockWidgetFeatures);
	dockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
	
	addDockWidget(Qt::LeftDockWidgetArea, dockWidget);
	dockWidget->setFloating(true);

	//---Timer---

	changeChecker_.setInterval( 500 );
	QObject::connect( &changeChecker_, SIGNAL( timeout() ), this, SLOT( updateTransferFunction() ) );

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

	buffer_ = Buffer1D::Ptr(new Buffer1D(4095, Interval(0.0f, 4095.0f)));	
		
	#ifdef TF_NDEBUG
		showMaximized();
	#endif
	#ifndef TF_NDEBUG
		show();
	#endif
	//show must be called before setting transfer function buffer
	mViewer->SetTransferFunctionBuffer(buffer_);
}

ViewerWindow::~ViewerWindow(){
}

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
	/*
	if(!buffer_ || (buffer_->Size() != editingSystem_->getDomain(TF_DIMENSION_1)))
	{
		buffer_ = Buffer1D::Ptr(new Buffer1D(editingSystem_->getDomain(TF_DIMENSION_1),
			Interval(0.0f, (float)editingSystem_->getDomain(TF_DIMENSION_1))));
	}
	
	bool tfUsed = editingSystem_->applyTransferFunction<Buffer1D::iterator>( buffer_->Begin(), buffer_->End());
	
	if(tfUsed)
	{
		mViewer->SetTransferFunctionBuffer(buffer_);

		QSize previewSize = editingSystem_->getPreviewSize();
		QImage thumbnailImage = mViewer->RenderThumbnailImage(previewSize);
		editingSystem_->setPreview(thumbnailImage);
	}
	*/

	if(fillBufferFromTF_(editingSystem_->getTransferFunction(), buffer_))
	{
		mViewer->SetTransferFunctionBuffer(buffer_);

		QSize previewSize = editingSystem_->getPreviewSize();
		QImage thumbnailImage = mViewer->RenderThumbnailImage(previewSize);
		editingSystem_->setPreview(thumbnailImage);
	}
}

bool ViewerWindow::fillBufferFromTF_(M4D::GUI::TFFunctionInterface::Const function, Buffer1D::Ptr& buffer){

	if(!function) return false;

	M4D::GUI::TF::Size domain = function.getDomain(TF_DIMENSION_1);
	if(!buffer || buffer->Size() != domain)
	{
		buffer = Buffer1D::Ptr(new Buffer1D(domain, Interval(0.0f, (float)domain)));
	}

	M4D::GUI::TF::Coordinates coords(1);
	M4D::GUI::TF::Color color;
	for(M4D::GUI::TF::Size i = 0; i < domain; ++i)
	{
		coords[0] = i;
		color = function.getRGBfColor(coords);

		(*buffer)[i] = Buffer1D::value_type(
			color.component1,
			color.component2,
			color.component3,
			color.alpha);
	}
	return true;
}

void ViewerWindow::updatePreview(M4D::GUI::TF::Size index){

	if(!fileLoaded_) return;	
	
	Buffer1D::Ptr buffer;
	if(fillBufferFromTF_(editingSystem_->getTransferFunction(index), buffer))
	{
		mViewer->SetTransferFunctionBuffer(buffer);

		QSize previewSize = editingSystem_->getPreviewSize();
		QImage thumbnailImage = mViewer->RenderThumbnailImage(previewSize);
		editingSystem_->setPreview(thumbnailImage, index);

		mViewer->SetTransferFunctionBuffer(buffer_);
	}
	
}

void ViewerWindow::updateTransferFunction(){

	M4D::Common::TimeStamp lastChange = editingSystem_->lastChange();
	if(lastChange_ != lastChange)
	{
		lastChange_ = lastChange;
		applyTransferFunction();
	}
}

void ViewerWindow::toggleInteractiveTransferFunction( bool aChecked )
{
	if ( aChecked )
	{
		D_PRINT( "Transfer function - interactive manipulation enabled" );
		changeChecker_.start();
	}
	else
	{
		D_PRINT( "Transfer function - interactive manipulation disabled" );
		changeChecker_.stop();
	}
}

void
ViewerWindow::updateToolbars()
{
	int rendererType = mViewer->GetRendererType();

	if(rendererType == M4D::GUI::rt3DGeneralSlices) rendererType = M4D::GUI::rt2DAlignedSlices;

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

	statusbar->showMessage("Loading data...");

	M4D::Imaging::AImage::Ptr image = M4D::Imaging::ImageFactory::LoadDumpedImage( path );
	mProdconn.PutDataset( image );
	fileLoaded_ = true;
	
	statusbar->showMessage("Computing histogram...");
	M4D::Common::Clock clock;

	Histogram::Ptr histogram;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		histogram = M4D::Imaging::CreateHistogramForImageRegion<Histogram, IMAGE_TYPE >( IMAGE_TYPE::Cast( *image ) );
	);
	
	TFHistogram::Ptr tfHistogram(new TFHistogram);
	for(Histogram::iterator it = histogram->Begin(); it != histogram->End(); ++it)
	{
		tfHistogram->add(*it);
	}
	editingSystem_->setDataStructure(std::vector<M4D::GUI::TF::Size>(1, tfHistogram->size()));
	editingSystem_->setHistogram(tfHistogram);	

	LOG( "Histogram computed in " << clock.SecondsPassed() );

	mViewer->ZoomFit();

	statusbar->showMessage("Applying transfer function...");
	applyTransferFunction();

	statusbar->clearMessage();
	changeChecker_.start();
}

void ViewerWindow::closeEvent(QCloseEvent *e){

	if(editingSystem_->close()) e->accept();
	else e->ignore();
}
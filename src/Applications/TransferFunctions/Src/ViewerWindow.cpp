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
	mViewer->SetColorTransformType( M4D::GUI::ctTransferFunction1D );
	action2D->setChecked(true);
	
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
	mViewer->SetRendererType( aRendererType );
}

void ViewerWindow::applyTransferFunction(){

	if(!fileLoaded_) return;

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
#include "GUI/utils/OGLDrawing.h"
#include "GUI/widgets/BasicSliceViewer.h"
#include "Imaging/ImageFactory.h"
namespace M4D
{
namespace GUI
{
namespace Viewer
{

BasicSliceViewer::BasicSliceViewer( QWidget *parent ) : 
	PredecessorType( parent ), _renderingMode( rmONE_DATASET ), _plane( XY_PLANE ), _lutWindow( 0.0f, 1.0f ), _interactionMode( imNONE ), _prepared( false )
{

}

BasicSliceViewer::~BasicSliceViewer()
{
	_shaderConfig.Finalize();
	cgDestroyContext(_cgContext);
}

void	
BasicSliceViewer::ZoomFit( ZoomType zoomType )
{
	_viewConfiguration = GetOptimalViewConfiguration(
			VectorPurgeDimension( _regionRealMin, _plane ), 
			VectorPurgeDimension( _regionRealMax, _plane ),
			Vector< uint32, 2 >( (uint32)width(), (uint32)height() ), 
			zoomType );
}

void
BasicSliceViewer::SetLUTWindow( Vector< float32, 2 > window )
{
	_lutWindow = window;
}

void
BasicSliceViewer::SetCurrentSlice( int32 slice )
{
	_currentSlice[ _plane ] = Max( Min( _regionMax[_plane], slice ), _regionMin[_plane] );
}

void	
BasicSliceViewer::initializeGL()
{
	//glEnable(GL_CULL_FACE);
	InitOpenGL();
	glClearColor(0,0,0.5f,1);
	//glEnable( GL_BLEND );
	//glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	//glDepthFunc(GL_LEQUAL);

	_cgContext = cgCreateContext();
	CheckForCgError("creating context ", _cgContext );

	_shaderConfig.Initialize( _cgContext, "LUT.cg", "SimpleBrightnessContrast3D" );
}

void	
BasicSliceViewer::initializeOverlayGL()
{

}

void	
BasicSliceViewer::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if( !IsDataPrepared() && !PrepareData() ) {
		return;
	}

	ZoomFit();

	switch ( _renderingMode )
	{
	case rmONE_DATASET:
		RenderOneDataset();
		break;
	default:
		ASSERT( false );
	}
}

void	
BasicSliceViewer::paintOverlayGL()
{

}

void	
BasicSliceViewer::resizeGL( int width, int height )
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
}

void	
BasicSliceViewer::resizeOverlayGL( int width, int height )
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
}

void	
BasicSliceViewer::mouseMoveEvent ( QMouseEvent * event )
{ 
	if( _interactionMode == imSETTING_LUT_WINDOW) {
		QPoint tmp = event->globalPos(); 
		int x = (tmp - _clickPosition).x();
		int y = (tmp - _clickPosition).y();
		SetLUTWindow( _oldLUTWindow + Vector< float32, 2 >( ((float32)x)/width(), ((float32)y)/height() ) );
		this->update();
	}
}

void	
BasicSliceViewer::mouseDoubleClickEvent ( QMouseEvent * event )
{
	if( event->button() == Qt::LeftButton ) {
		_plane = NextCartesianPlane( _plane );
		this->update();
		return;
	}
}

void	
BasicSliceViewer::mousePressEvent ( QMouseEvent * event )
{ 	
	_clickPosition = event->globalPos();

	if( event->button() == Qt::RightButton ) {
		_interactionMode = imSETTING_LUT_WINDOW;
		_oldLUTWindow = GetLUTWindow();
		this->update();
		return;
	}
}

void	
BasicSliceViewer::mouseReleaseEvent ( QMouseEvent * event )
{ 
	_interactionMode = imNONE; 
}

void	
BasicSliceViewer::wheelEvent ( QWheelEvent * event )
{
	int numDegrees = event->delta() / 8;
	int numSteps = numDegrees / 15;
	if (event->orientation() == Qt::Horizontal) {
		SetCurrentSlice( _currentSlice[ _plane ] -= numSteps );
	} else {
		SetCurrentSlice( _currentSlice[ _plane ] += numSteps );
	}
	event->accept();
	this->update();
}



bool
BasicSliceViewer::IsDataPrepared()
{
	return _prepared;
}

bool
BasicSliceViewer::PrepareData()
{

/*	if ( !_image )
		return false;

	_textureData = CreateTextureFromImage( *_image->GetAImageRegion() );
	_regionMin = M4D::Imaging::AImageDim<3>::Cast( boost::static_pointer_cast< M4D::Imaging::ADataset > ( _image ) )->GetRealMinimum();
	_regionMax = M4D::Imaging::AImageDim<3>::Cast( boost::static_pointer_cast< M4D::Imaging::ADataset > ( _image ) )->GetRealMaximum();

	return true;*/

	try {
		TryGetAndLockAllInputs();
	} catch (...) {
		return false;
	}

	_regionMin = M4D::Imaging::AImageDim<3>::Cast( _inputDatasets[0] )->GetMinimum();
	_regionMax = M4D::Imaging::AImageDim<3>::Cast( _inputDatasets[0] )->GetMaximum();
	_regionRealMin = M4D::Imaging::AImageDim<3>::Cast( _inputDatasets[0] )->GetRealMinimum();
	_regionRealMax = M4D::Imaging::AImageDim<3>::Cast( _inputDatasets[0] )->GetRealMaximum();
	_elementExtents = M4D::Imaging::AImageDim<3>::Cast( _inputDatasets[0] )->GetElementExtents();

	_currentSlice = _regionMin;

	/*M4D::Imaging::Image<uint16,3>::Ptr image = M4D::Imaging::Image<uint16,3>::Cast( _inputDatasets[0] );
	M4D::Imaging::ImageFactory::DumpImage( "pom.dump", *image );
	*/



	_textureData = CreateTextureFromImage( *(M4D::Imaging::AImage::Cast( _inputDatasets[0] )->GetAImageRegion()) ) ;

	ReleaseAllInputs();

	_prepared = true;
	return true;
}

void	
BasicSliceViewer::RenderOneDataset()
{
	glBindTexture( GL_TEXTURE_1D, 0 );
	glBindTexture( GL_TEXTURE_2D, 0 );
	glBindTexture( GL_TEXTURE_3D, 0 );
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_1D);


	_shaderConfig.textureName = _textureData->GetTextureGLID();
	_shaderConfig.brightnessContrast = _lutWindow;
	_shaderConfig.Enable();
	
	CheckForCgError("Check before drawing ", _cgContext );
	//M4D::GLDrawTexturedQuad( _textureData->GetMinimum3D(), _textureData->GetMaximum3D() );
	SetToViewConfiguration2D( _viewConfiguration );
	M4D::GLDrawVolumeSlice( _textureData->GetMinimum3D(), _textureData->GetMaximum3D(), (float32)_currentSlice[ _plane ] * _elementExtents[_plane], _plane );
	
	_shaderConfig.Disable();
	
	glFlush();
}

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



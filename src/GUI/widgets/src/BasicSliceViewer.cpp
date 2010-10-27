#ifdef USE_CG

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
	PredecessorType( parent ), _renderingMode( rmONE_DATASET ), _interactionMode( imNONE ), _prepared( false ), mSaveFile( false ), mSaveCycle( false )
{

}

BasicSliceViewer::~BasicSliceViewer()
{
	_renderer.Finalize();

	/*GL_CHECKED_CALL( glDeleteFramebuffersEXT( 1, &mFrameBufferObject ) );
	GL_CHECKED_CALL( glDeleteTextures( 1, &mColorTexture ) );
	GL_CHECKED_CALL( glDeleteRenderbuffersEXT( 1, &mDepthBuffer ) );*/
}

void	
BasicSliceViewer::ZoomFit( ZoomType zoomType )
{
	_renderer.GetSliceViewConfig().viewConfiguration = GetOptimalViewConfiguration(
			VectorPurgeDimension( _regionRealMin, _renderer.GetSliceViewConfig().plane ), 
			VectorPurgeDimension( _regionRealMax, _renderer.GetSliceViewConfig().plane ),
			Vector< uint32, 2 >( (uint32)width(), (uint32)height() ), 
			zoomType );
}

void
BasicSliceViewer::SetLUTWindow( Vector< float32, 2 > window )
{
	_lutWindow = window;
	_renderer.SetLUTWindow( _lutWindow );
}


void
BasicSliceViewer::SetTransferFunctionBuffer( TransferFunctionBuffer1D::Ptr aTFunctionBuffer )
{
	if ( !aTFunctionBuffer ) {
		_THROW_ ErrorHandling::EBadParameter();
	}
	mTFunctionBuffer = aTFunctionBuffer;

	mTransferFunctionTexture = CreateGLTransferFunctionBuffer1D( *aTFunctionBuffer );

	_renderer.SetTransferFunction( mTransferFunctionTexture );

	update();
}

void
BasicSliceViewer::SetCurrentSlice( int32 slice )
{
	_renderer.GetSliceViewConfig().currentSlice[ _renderer.GetSliceViewConfig().plane ] = Max( Min( _regionMax[_renderer.GetSliceViewConfig().plane], slice ), _regionMin[_renderer.GetSliceViewConfig().plane] );
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


	_renderer.Initialize();
	mFrameBufferObject.Initialize( width(), height() );


	//******************************************************************
	/*QGLFormat indirectRenderingFormat( format() );
	indirectRenderingFormat.setDirectRendering( false );
	mOtherContext = new QGLContext( indirectRenderingFormat );
	bool res = mOtherContext->create( context() );
	ASSERT( res );
	ASSERT( mOtherContext->isValid() );

	mRenderingThread.SetContext( *mOtherContext );
	mRenderingThread.start();*/
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
//******************************************************* TODO delete

	/*if (mSaveCycle) {
		static size_t counter = 0;
		ResetView();
		
		size_t stepCount = 125;
		float rotStep = PIx2 / stepCount;
		for( size_t i = 0; i < stepCount; ++ i ) {
			_renderer.FineRender();
			mFrameBufferObject.Bind();
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			_renderer.Render();
			mFrameBufferObject.Unbind();

			Vector2u size = mFrameBufferObject.GetSize();
			std::string filename = TO_STRING( "output_" << counter << "_" << i << ".png" );
			SaveTextureToImageFile( size[0], size[1], mFrameBufferObject.GetColorBuffer(), filename, true );
			mSaveFile = false;
			
			_renderer.GetViewConfig3D().camera.YawAround( rotStep );
			LOG( i << " / " << stepCount );
		}
		++counter;
		mSaveCycle = false;
	}*/
//*******************************************************

	mFrameBufferObject.Bind();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	_renderer.Render();
	mFrameBufferObject.Unbind();


#ifdef USE_DEVIL
	if (mSaveFile) {
		Vector2u size = mFrameBufferObject.GetSize();
		SaveTextureToImageFile( size[0], size[1], mFrameBufferObject.GetColorBuffer(), "output.png", true );
		mSaveFile = false;
	}
#endif /*USE_DEVIL*/
// ***************************************************************************************************

	mFrameBufferObject.Render();	

	M4D::CheckForGLError( "OGL error : " );
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

	float x = (float)width / height;
	_renderer.GetViewConfig3D().camera.SetAspectRatio( x );


	mFrameBufferObject.Resize( width, height );
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
	QPoint tmp = event->globalPos(); 
	if( _interactionMode == imSETTING_LUT_WINDOW) {
		int x = (tmp - _clickPosition).x();
		int y = (tmp - _clickPosition).y();
		SetLUTWindow( _oldLUTWindow + Vector< float32, 2 >( 3000*((float32)x)/width(), 3000*((float32)y)/height() ) );
		this->update();
	}
	if( _interactionMode == imORBIT_CAMERA) {
		int x = (tmp - mLastPoint).x();
		int y = (tmp - mLastPoint).y();
		mLastPoint = event->globalPos();
		_renderer.GetViewConfig3D().camera.YawAround( x * -0.05f );
		_renderer.GetViewConfig3D().camera.PitchAround( y * -0.05f );
		this->update();
	}

}

void	
BasicSliceViewer::mouseDoubleClickEvent ( QMouseEvent * event )
{
	if( event->button() == Qt::LeftButton ) {
		_renderer.GetSliceViewConfig().plane = NextCartesianPlane( _renderer.GetSliceViewConfig().plane );
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

	if( event->button() == Qt::LeftButton ) {
		_interactionMode = imORBIT_CAMERA;
		mLastPoint = _clickPosition;
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
	float dollyRatio = 1.1f;
	if ( event->delta() > 0 ) {
		dollyRatio = 1.0f/dollyRatio;
	}
	switch ( _renderer.GetRendererType() ) {
	case rt2DAlignedSlices:
			SetCurrentSlice( _renderer.GetSliceViewConfig().currentSlice[ _renderer.GetSliceViewConfig().plane ] += numSteps );
		break;
	case rt3DGeneralSlices:
		return;
	case rt3D:
		DollyCamera( _renderer.GetViewConfig3D().camera, dollyRatio );
		break;
	default:
		ASSERT( false );
	};
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

	_regionMin = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetMinimum();
	_regionMax = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetMaximum();
	_regionRealMin = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetRealMinimum();
	_regionRealMax = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetRealMaximum();
	_elementExtents = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetElementExtents();

	_renderer.GetSliceViewConfig().currentSlice = _regionMin;

	/*M4D::Imaging::Image<uint16,3>::Ptr image = M4D::Imaging::Image<uint16,3>::Cast( mInputDatasets[0] );
	M4D::Imaging::ImageFactory::DumpImage( "pom.dump", *image );
	*/



	_textureData = CreateTextureFromImage( *(M4D::Imaging::AImage::Cast( mInputDatasets[0] )->GetAImageRegion()), true ) ;

	ReleaseAllInputs();


	_renderer.SetImageData( _textureData );

	_prepared = true;
	return true;
}

void	
BasicSliceViewer::RenderOneDataset()
{
	/*glBindTexture( GL_TEXTURE_1D, 0 );
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
	
	glFlush();*/
}

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*USE_CG*/

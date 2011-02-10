#ifdef USE_CG

#include "GUI/utils/OGLDrawing.h"
#include "GUI/utils/QtM4DTools.h"
#include "common/MathTools.h"
#include "GUI/widgets/BasicSliceViewer.h"
#include "Imaging/ImageFactory.h"
#include "GUI/utils/CameraManipulator.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

BasicSliceViewer::BasicSliceViewer( QWidget *parent ) : 
	PredecessorType( parent ), _renderingMode( rmONE_DATASET ), _prepared( false ), mSaveFile( false ), mSaveCycle( false )
{
     /*QState *s1 = new QState();
     QState *s2 = new QState();
     QState *s3 = new QState();
     QAbstractTransition *t1 = new IntCheckedSignalTransition( this, SIGNAL( RendererTypeChanged(int) ), rt3D );
     t1->setTargetState(s2);
     s1->addTransition(t1);
     s2->addTransition(button, SIGNAL(clicked()), s3);
     s3->addTransition(button, SIGNAL(clicked()), s1);
     mStateMachine.addState(s1);
     mStateMachine.addState(s2);
     mStateMachine.addState(s3);
     mStateMachine.setInitialState(s1);*/
	mSliceRenderConfig.colorTransform = M4D::GUI::Renderer::SliceRenderer::ctLUTWindow;
	mSliceRenderConfig.plane = XY_PLANE;

	mVolumeRenderConfig.colorTransform = M4D::GUI::Renderer::VolumeRenderer::ctMaxIntensityProjection;
	mVolumeRenderConfig.sampleCount = 200;
	mVolumeRenderConfig.shadingEnabled = true;
	mVolumeRenderConfig.jitterEnabled = true;

	mUserEventHandlers[ rt2DAlignedSlices ] = IUserEvents::Ptr( new SliceViewConfigManipulator( &(mSliceRenderConfig.viewConfig), &(mSliceRenderConfig.currentSlice), &(mSliceRenderConfig.plane) ) );
	mUserEventHandlers[ rt3D ] = IUserEvents::Ptr( new CameraManipulator( &(mVolumeRenderConfig.camera) ) );

	mCurrentEventHandler = mUserEventHandlers[ rt2DAlignedSlices ].get();
}

BasicSliceViewer::~BasicSliceViewer()
{
	mSliceRenderer.Finalize();
	mVolumeRenderer.Finalize();
}

void	
BasicSliceViewer::ZoomFit( ZoomType zoomType )
{
	mSliceRenderConfig.viewConfig = GetOptimalViewConfiguration(
			VectorPurgeDimension( _regionRealMin, mSliceRenderConfig.plane ), 
			VectorPurgeDimension( _regionRealMax, mSliceRenderConfig.plane ),
			Vector< uint32, 2 >( (uint32)width(), (uint32)height() ), 
			zoomType );
}

void
BasicSliceViewer::SetLUTWindow( Vector2f window )
{
	mSliceRenderConfig.lutWindow = window;
	mVolumeRenderConfig.lutWindow = window;
}


void
BasicSliceViewer::SetTransferFunctionBuffer( TransferFunctionBuffer1D::Ptr aTFunctionBuffer )
{
	if ( !aTFunctionBuffer ) {
		_THROW_ ErrorHandling::EBadParameter();
	}
	mTFunctionBuffer = aTFunctionBuffer;

	mTransferFunctionTexture = CreateGLTransferFunctionBuffer1D( *aTFunctionBuffer );

	mSliceRenderConfig.transferFunction = mTransferFunctionTexture.get();
	mVolumeRenderConfig.transferFunction = mTransferFunctionTexture.get();

	update();
}

void
BasicSliceViewer::SetCurrentSlice( int32 slice )
{
	mSliceRenderConfig.currentSlice[ mSliceRenderConfig.plane ] = Max( Min( _regionMax[mSliceRenderConfig.plane]-1, slice ), _regionMin[mSliceRenderConfig.plane] );
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


	mSliceRenderer.Initialize();
	mVolumeRenderer.Initialize();
	//_renderer.Initialize();
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

	//mDummyGLWidget = new QGLWidget( this, this );
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
	
	switch ( mRendererType ) {
	case rt3D:
		mVolumeRenderer.Render( mVolumeRenderConfig );
		break;
	case rt2DAlignedSlices:
		mSliceRenderer.Render( mSliceRenderConfig );
		break;
	default:
		ASSERT( false );
	}
	
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
	mVolumeRenderConfig.camera.SetAspectRatio( x );


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
	if ( mCurrentEventHandler && mCurrentEventHandler->mouseMoveEvent( this->size(), event ) ) {
		this->update();
		return;
	}
	/*SliceViewConfig & sliceViewConfig = _renderer.GetSliceViewConfig();
	Vector2f coords = GetRealCoordinatesFromScreen( 
						QPointFToVector2f(event->posF()), 
						QSizeToVector2u( size() ), 
						sliceViewConfig.viewConfiguration 
						);
	QString info;
	//info += VectorToQString( coords );
	//info += "DatasetCoords : ";

	Vector2f extents = VectorPurgeDimension( _elementExtents, sliceViewConfig.plane );
	Vector3i dataCoords = VectorInsertDimension( Round<2>( VectorMemberDivision( coords, extents ) ), sliceViewConfig.currentSlice[sliceViewConfig.plane], sliceViewConfig.plane );

	info += VectorToQString( dataCoords ); 
	info += " : ";
	info += GetVoxelInfo( dataCoords );

	emit MouseInfoUpdate( info );*/
}

void	
BasicSliceViewer::mouseDoubleClickEvent ( QMouseEvent * event )
{
	if ( mCurrentEventHandler && mCurrentEventHandler->mouseDoubleClickEvent( this->size(), event ) ) {
		this->update();
		return;
	}
}

void	
BasicSliceViewer::mousePressEvent ( QMouseEvent * event )
{ 	
	if ( mCurrentEventHandler && mCurrentEventHandler->mousePressEvent( this->size(), event ) ) {
		this->update();
		return;
	}
}

void	
BasicSliceViewer::mouseReleaseEvent ( QMouseEvent * event )
{ 
	if ( mCurrentEventHandler && mCurrentEventHandler->mouseReleaseEvent( this->size(), event ) ) {
		this->update();
		return;
	}
}

void	
BasicSliceViewer::wheelEvent ( QWheelEvent * event )
{
	if ( mCurrentEventHandler && mCurrentEventHandler->wheelEvent( this->size(), event ) ) {
		this->update();
		return;
	}
}



bool
BasicSliceViewer::IsDataPrepared()
{
	return _prepared;
}

bool
BasicSliceViewer::PrepareData()
{
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

	mSliceRenderConfig.currentSlice = _regionMin;

	_textureData = CreateTextureFromImage( *(M4D::Imaging::AImage::Cast( mInputDatasets[0] )->GetAImageRegion()), true ) ;

	ReleaseAllInputs();


	mSliceRenderConfig.imageData = &(_textureData->GetDimensionedInterface<3>());
	mVolumeRenderConfig.imageData = &(_textureData->GetDimensionedInterface<3>());

	mVolumeRenderConfig.camera.SetTargetPosition( 0.5f * (_textureData->GetDimensionedInterface< 3 >().GetMaximum() + _textureData->GetDimensionedInterface< 3 >().GetMinimum()) );
	mVolumeRenderConfig.camera.SetFieldOfView( 45.0f );
	mVolumeRenderConfig.camera.SetEyePosition( Vector3f( 0.0f, 0.0f, 750.0f ) );
	ResetView();

	_prepared = true;
	return true;
}

QString
BasicSliceViewer::GetVoxelInfo( Vector3i aDataCoords )
{
	try {
		TryGetAndLockAllInputs();
	} catch (...) {
		return QString("NONE");
	}
	M4D::Imaging::AImage::ConstPtr image = M4D::Imaging::AImage::Cast( mInputDatasets[0] );
	QString result;

	try {
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::ConstPtr typedImage = IMAGE_TYPE::Cast( image );
			result = QString::number( typedImage->GetElement( aDataCoords ) );
			);

	} catch (...) {
		result = QString("NONE");
	}
	ReleaseAllInputs();
	return result;
}

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*USE_CG*/

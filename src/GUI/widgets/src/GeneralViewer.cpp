#ifdef USE_CG

#include "GUI/utils/OGLDrawing.h"
#include "GUI/utils/QtM4DTools.h"
#include "common/MathTools.h"
#include "GUI/widgets/GeneralViewer.h"
#include "Imaging/ImageFactory.h"
#include "GUI/utils/CameraManipulator.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

GeneralViewer::GeneralViewer( QWidget *parent ): PredecessorType( parent ), _prepared( false )
{
	ViewerState * state = new ViewerState;

	state->mSliceRenderConfig.colorTransform = M4D::GUI::Renderer::SliceRenderer::ctLUTWindow;
	state->mSliceRenderConfig.plane = XY_PLANE;

	state->mVolumeRenderConfig.colorTransform = M4D::GUI::Renderer::VolumeRenderer::ctMaxIntensityProjection;
	state->mVolumeRenderConfig.sampleCount = 200;
	state->mVolumeRenderConfig.shadingEnabled = true;
	state->mVolumeRenderConfig.jitterEnabled = true;

	state->viewerWindow = this;

	state->backgroundColor = QColor( 20, 10, 90);

	state->availableViewTypes = 7;
	state->viewType = vt2DAlignedSlices;

	mViewerState = BaseViewerState::Ptr( state );
}


void
GeneralViewer::SetLUTWindow( Vector2f window )
{
	getViewerState().mSliceRenderConfig.lutWindow = window;
	getViewerState().mVolumeRenderConfig.lutWindow = window;
}


void
GeneralViewer::SetTransferFunctionBuffer( TransferFunctionBuffer1D::Ptr aTFunctionBuffer )
{
	if ( !aTFunctionBuffer ) {
		_THROW_ ErrorHandling::EBadParameter();
	}
	getViewerState().mTFunctionBuffer = aTFunctionBuffer;
	
	makeCurrent();
	getViewerState().mTransferFunctionTexture = CreateGLTransferFunctionBuffer1D( *aTFunctionBuffer );
	doneCurrent();

	getViewerState().mSliceRenderConfig.transferFunction = getViewerState().mTransferFunctionTexture.get();
	getViewerState().mVolumeRenderConfig.transferFunction = getViewerState().mTransferFunctionTexture.get();

	update();
}

void
GeneralViewer::SetCurrentSlice( int32 slice )
{
	CartesianPlanes plane = getViewerState().mSliceRenderConfig.plane;
	getViewerState().mSliceRenderConfig.currentSlice[ plane ] = Max( 
								Min( getViewerState()._regionMax[plane]-1, slice ), 
								getViewerState()._regionMin[plane] );
}


//********************************************************************************

void
GeneralViewer::initializeRenderingEnvironment()
{
	getViewerState().mSliceRenderer.Initialize();
	getViewerState().mVolumeRenderer.Initialize();
}

bool
GeneralViewer::preparedForRendering()
{
	if( !IsDataPrepared() && !PrepareData() ) {
		return false;
	}
	return true;
}

void
GeneralViewer::prepareForRenderingStep()
{

}

void
GeneralViewer::render()
{

}

void
GeneralViewer::finalizeAfterRenderingStep()
{

}

//***********************************************************************************

bool
GeneralViewer::IsDataPrepared()
{
	return _prepared;
}

bool
GeneralViewer::PrepareData()
{
	try {
		TryGetAndLockAllInputs();
	} catch (...) {
		return false;
	}

	getViewerState()._regionMin = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetMinimum();
	getViewerState()._regionMax = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetMaximum();
	getViewerState()._regionRealMin = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetRealMinimum();
	getViewerState()._regionRealMax = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetRealMaximum();
	getViewerState()._elementExtents = M4D::Imaging::AImageDim<3>::Cast( mInputDatasets[0] )->GetElementExtents();

	getViewerState().mSliceRenderConfig.currentSlice = getViewerState()._regionMin;

	getViewerState()._textureData = CreateTextureFromImage( *(M4D::Imaging::AImage::Cast( mInputDatasets[0] )->GetAImageRegion()), true ) ;

	ReleaseAllInputs();


	getViewerState().mSliceRenderConfig.imageData = &(getViewerState()._textureData->GetDimensionedInterface<3>());
	getViewerState().mVolumeRenderConfig.imageData = &(getViewerState()._textureData->GetDimensionedInterface<3>());

	getViewerState().mVolumeRenderConfig.camera.SetTargetPosition( 0.5f * (getViewerState()._textureData->GetDimensionedInterface< 3 >().GetMaximum() + getViewerState()._textureData->GetDimensionedInterface< 3 >().GetMinimum()) );
	getViewerState().mVolumeRenderConfig.camera.SetFieldOfView( 45.0f );
	getViewerState().mVolumeRenderConfig.camera.SetEyePosition( Vector3f( 0.0f, 0.0f, 750.0f ) );
	ResetView();

	_prepared = true;
	return true;
}


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*USE_CG*/

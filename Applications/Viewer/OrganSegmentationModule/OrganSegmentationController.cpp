#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include <algorithm>

OrganSegmentationController::OrganSegmentationController(): mBrushValue( 255 )
{

	
}

OrganSegmentationController::~OrganSegmentationController()
{

}

bool
OrganSegmentationController::mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMaskDrawingController ) {
		if ( mMaskDrawingController->mouseMoveEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseMoveEvent ( aViewerState, aEventInfo );
}

bool	
OrganSegmentationController::mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMaskDrawingController ) {
		if ( mMaskDrawingController->mouseDoubleClickEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseDoubleClickEvent ( aViewerState, aEventInfo );
}

bool
OrganSegmentationController::mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMaskDrawingController ) {
		if ( mMaskDrawingController->mousePressEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mousePressEvent ( aViewerState, aEventInfo );
}

bool
OrganSegmentationController::mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMaskDrawingController ) {
		if ( mMaskDrawingController->mouseReleaseEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseReleaseEvent ( aViewerState, aEventInfo );
}

bool
OrganSegmentationController::wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event )
{
	if ( mMaskDrawingController ) {
		if ( mMaskDrawingController->wheelEvent( aViewerState, event ) ) {
			return true;
		}
	}
	return ControllerPredecessor::wheelEvent ( aViewerState, event );
}

unsigned
OrganSegmentationController::getAvailableViewTypes()const
{

}

void
OrganSegmentationController::render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane )
{
	
}
void
OrganSegmentationController::preRender3D()
{
	if( !mOverlay ) {
		render3D();
	}
}

void
OrganSegmentationController::postRender3D()
{
	if( mOverlay ) {
		render3D();
	}
}

void
OrganSegmentationController::render3D()
{
	
}

void
OrganSegmentationController::toggleMaskDrawing( bool aToggle )
{
	if( aToggle ) {
		mMaskDrawingController = MaskDrawingMouseController::Ptr( new MaskDrawingMouseController( mMask, 255 ) );
	} else {
		mMaskDrawingController = MaskDrawingMouseController::Ptr();
	}
}

void
OrganSegmentationController::toggleBiMaskDrawing( bool aToggle )
{
	if( aToggle ) {
		mMaskDrawingController = MaskDrawingMouseController::Ptr( new MaskDrawingMouseController( mMask, mBrushValue ) );
	} else {
		mMaskDrawingController = MaskDrawingMouseController::Ptr();
	}
}

void
OrganSegmentationController::changeMarkerType( bool aForeground )
{
	mBrushValue = aForeground ? 255 : 100;
	if( mMaskDrawingController ) {
			mMaskDrawingController->setBrushValue( mBrushValue );
	}
}




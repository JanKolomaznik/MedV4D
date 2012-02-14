#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include <algorithm>

OrganSegmentationController::OrganSegmentationController()
{

	
}

OrganSegmentationController::~OrganSegmentationController()
{

}

bool
OrganSegmentationController::mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	return ControllerPredecessor::mouseMoveEvent ( aViewerState, aEventInfo );
}

bool	
OrganSegmentationController::mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	
	return ControllerPredecessor::mouseDoubleClickEvent ( aViewerState, aEventInfo );
}

bool
OrganSegmentationController::mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	return ControllerPredecessor::mousePressEvent ( aViewerState, aEventInfo );
}

bool
OrganSegmentationController::mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	return ControllerPredecessor::mouseReleaseEvent ( aViewerState, aEventInfo );
}

bool
OrganSegmentationController::wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event )
{
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




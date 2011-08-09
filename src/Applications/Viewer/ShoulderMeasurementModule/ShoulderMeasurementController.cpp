#include "ShoulderMeasurementModule/ShoulderMeasurementController.hpp"

class PointGroupPrimitiveController: public TemplatedPrimitiveCreationEventController< M4D::Point3Df >
{
public:
	PointGroupPrimitiveController( PointSet &aPrimitives, size_t aMaxPointCount ): mPrimitives( aPrimitives ), mMaxPointCount( aMaxPointCount ) {}

protected:

	virtual M4D::Point3Df *
	createPrimitive( const M4D::Point3Df & aPrimitive )
	{
		if ( mPrimitives.size() == mMaxPointCount ) {
			return NULL;
		}
		mPrimitives.push_back( aPrimitive );
		return &(mPrimitives[mPrimitives.size()-1]);
	}

	virtual void
	primitiveFinished( M4D::Point3Df *aPrimitive )
	{
	}

	virtual void
	disposePrimitive( M4D::Point3Df *aPrimitive )
	{
		mPrimitives.resize( mPrimitives.size()-1 );
	}

	PointSet &mPrimitives;
	size_t mMaxPointCount;
};



ShoulderMeasurementController::ShoulderMeasurementController()
{
	mMeasurementHandlers[mmHUMERAL_HEAD] = APrimitiveCreationEventController::Ptr( new PointGroupPrimitiveController(mPoints, 6) );

}

ShoulderMeasurementController::~ShoulderMeasurementController()
{

}

bool
ShoulderMeasurementController::mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->mouseMoveEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseMoveEvent ( aViewerState, aEventInfo );
}
bool	
ShoulderMeasurementController::mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->mouseDoubleClickEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
		return true; //TODO - prevent plane changing
	}
	return ControllerPredecessor::mouseDoubleClickEvent ( aViewerState, aEventInfo );
}

bool
ShoulderMeasurementController::mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->mousePressEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mousePressEvent ( aViewerState, aEventInfo );
}

bool
ShoulderMeasurementController::mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->mouseReleaseEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseReleaseEvent ( aViewerState, aEventInfo );
}

bool
ShoulderMeasurementController::wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->wheelEvent( aViewerState, event ) ) {
			return true;
		}
	}
	return ControllerPredecessor::wheelEvent ( aViewerState, event );
}

unsigned
ShoulderMeasurementController::getAvailableViewTypes()const
{

}

void
ShoulderMeasurementController::render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane )
{

}

void
ShoulderMeasurementController::preRender3D()
{

}

void
ShoulderMeasurementController::postRender3D()
{

}

void
ShoulderMeasurementController::render3D()
{

}

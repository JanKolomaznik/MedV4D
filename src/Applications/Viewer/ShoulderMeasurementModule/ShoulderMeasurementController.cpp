#include "ShoulderMeasurementModule/ShoulderMeasurementController.hpp"
#include <algorithm>

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
	mMeasurementHandlers[mmHUMERAL_HEAD] = APrimitiveCreationEventController::Ptr( new PointGroupPrimitiveController(mHumeralHeadPoints, 6) );

	mMeasurementHandlers[mmPROXIMAL_SHAFT] = APrimitiveCreationEventController::Ptr( new PointGroupPrimitiveController(mProximalShaftPoints, 20) );

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
	GL_CHECKED_CALL( glPushAttrib( GL_ALL_ATTRIB_BITS ) );

	GL_CHECKED_CALL( glEnable( GL_POINT_SMOOTH ) );
	GL_CHECKED_CALL( glEnable( GL_BLEND ) );
	GL_CHECKED_CALL( glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) );
	GL_CHECKED_CALL( glPointSize( 4.0f ) );

	GL_CHECKED_CALL( glDisable( GL_DEPTH_TEST ) );
	
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );
	drawPointSet2D( mHumeralHeadPoints.begin(), mHumeralHeadPoints.end(), aInterval, aPlane );

	GL_CHECKED_CALL( glColor4f( 1.0f, 0.75f, 0.0f, 1.0f ) );
	drawPointSet2D( mProximalShaftPoints.begin(), mProximalShaftPoints.end(), aInterval, aPlane );

	GL_CHECKED_CALL( glPopAttrib() );
}

void
ShoulderMeasurementController::preRender3D()
{
	if( !mOverlay ) {
		render3D();
	}
}

void
ShoulderMeasurementController::postRender3D()
{
	if( mOverlay ) {
		render3D();
	}
}

void
ShoulderMeasurementController::render3D()
{
	GL_CHECKED_CALL( glPushAttrib( GL_ALL_ATTRIB_BITS ) );

	GL_CHECKED_CALL( glEnable( GL_POINT_SMOOTH ) );
	GL_CHECKED_CALL( glEnable( GL_BLEND ) );
	GL_CHECKED_CALL( glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) );
	GL_CHECKED_CALL( glPointSize( 4.0f ) );

	GL_CHECKED_CALL( glDisable( GL_LIGHTING ) );

	//**************************************
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );
	M4D::drawPointSet( mHumeralHeadPoints.begin(), mHumeralHeadPoints.end() );

	if ( mHeadMeasurementData.available ) {
		GL_CHECKED_CALL( glColor4f( 0.0f, 1.0f, 0.0f, 1.0f ) );
		M4D::drawGrid( mHeadMeasurementData.point, mHeadMeasurementData.vDirection, mHeadMeasurementData.wDirection, 80.0f, 80.0f, 5.0f );

		M4D::drawStippledLine( mHeadMeasurementData.point + mHeadMeasurementData.normal * 70.0f, mHeadMeasurementData.point - mHeadMeasurementData.normal * 70.0f );
	}

	//**************************************
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.75f, 0.0f, 1.0f ) );
	M4D::drawPointSet( mProximalShaftPoints.begin(), mProximalShaftPoints.end() );

	if ( mProximalShaftMeasurementData.available ) {
		GL_CHECKED_CALL( glColor4f( 0.0f, 1.0f, 1.0f, 1.0f ) );
		M4D::GLDrawBoundingBox( mProximalShaftMeasurementData.minimum, mProximalShaftMeasurementData.maximum );

		GL_CHECKED_CALL( glEnable( GL_LIGHTING ) );
		M4D::drawCylinder( mProximalShaftMeasurementData.point, mProximalShaftMeasurementData.direction, mProximalShaftMeasurementData.radius, mProximalShaftMeasurementData.height );

		//M4D::drawStippledLine( mHeadMeasurementData.point + mHeadMeasurementData.normal * 70.0f, mHeadMeasurementData.point - mHeadMeasurementData.normal * 70.0f );
	}
	GL_CHECKED_CALL( glPopAttrib() );
}



void
ShoulderMeasurementController::analyseHumeralHead()
{
	if ( mHumeralHeadPoints.size() < 3 ) {
		return;
	}

	getHeadMeasurementData( mHumeralHeadPoints, mHeadMeasurementData );

	emit updateRequest();
}

void
ShoulderMeasurementController::analyseProximalShaftOfHumerus()
{
	if ( mProximalShaftPoints.size() < 10 ) {
		return;
	}

	getProximalShaftMeasurementData( mProximalShaftPoints, mProximalShaftMeasurementData );

	emit updateRequest();
}

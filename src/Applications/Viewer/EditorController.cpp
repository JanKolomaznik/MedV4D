#include "EditorController.hpp"


EditorController::EditorController()
{
	mVectorEditorInteractionButton = Qt::LeftButton;

	mOverlay = false;
}

/*bool
mouseMoveEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

bool	
mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );*/

bool
EditorController::mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QMouseEvent * event )
{
	D_PRINT( "mousePressEvent" );
	M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );

	CartesianPlanes plane = state.mSliceRenderConfig.plane;
	Vector3f realSlices = state.mSliceRenderConfig.getCurrentRealSlice();
	if ( event->button() == mVectorEditorInteractionButton 
		&& state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) {
		Vector2f pos = GetRealCoordinatesFromScreen( 
					Vector2f( event->posF().x(), event->posF().y() ), 
					state.mWindowSize, 
					state.mSliceRenderConfig.viewConfig 
					);
			
		
		mPoints.addPoint( VectorInsertDimension( pos, realSlices[plane], plane ) );
		state.viewerWindow->update();
		return true;
	}
	
	D_PRINT( "mousePressEvent not handled" );
	return Predecessor1::mousePressEvent ( aViewerState, event );
}

/*bool
mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

bool
wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event );*/

unsigned
EditorController::getAvailableViewTypes()const
{
	return M4D::GUI::Viewer::vt3D | M4D::GUI::Viewer::vt2DAlignedSlices;
}

void
EditorController::render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane )
{
	GL_CHECKED_CALL( glPushAttrib( GL_ALL_ATTRIB_BITS ) );

	GL_CHECKED_CALL( glEnable( GL_POINT_SMOOTH ) );
	GL_CHECKED_CALL( glEnable( GL_BLEND ) );
	GL_CHECKED_CALL( glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) );
	GL_CHECKED_CALL( glPointSize( 4.0f ) );

	GL_CHECKED_CALL( glDisable( GL_DEPTH_TEST ) );
	
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );

	glBegin( GL_POINTS );
		for( size_t i = 0; i < mPoints.mPoints.size(); ++i ) {
			if ( IntervalTest( aInterval[0], aInterval[1], mPoints.mPoints[i][aPlane] ) ) { 
				//LOG( "rendering point " << mPoints.mPoints[i] );
				M4D::GLVertexVector( VectorPurgeDimension( mPoints.mPoints[i], aPlane ) );
			}
		}
		//std::for_each( mPoints.mPoints.begin(), mPoints.mPoints.end(), M4D::GLVertexVector );
	glEnd();

	GL_CHECKED_CALL( glPopAttrib() );
}

void
EditorController::preRender3D()
{
	if( !mOverlay ) {
		render3D();
	}
}

void
EditorController::postRender3D()
{
	if( mOverlay ) {
		render3D();
	}
}

void
EditorController::render3D()
{
	GL_CHECKED_CALL( glPushAttrib( GL_ALL_ATTRIB_BITS ) );

	GL_CHECKED_CALL( glEnable( GL_POINT_SMOOTH ) );
	GL_CHECKED_CALL( glEnable( GL_BLEND ) );
	GL_CHECKED_CALL( glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) );
	GL_CHECKED_CALL( glPointSize( 4.0f ) );

	
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );

	glBegin( GL_POINTS );
		for( size_t i = 0; i < mPoints.mPoints.size(); ++i ) {
			M4D::GLVertexVector( mPoints.mPoints[i] );
		}
		//std::for_each( mPoints.mPoints.begin(), mPoints.mPoints.end(), M4D::GLVertexVector );
	glEnd();

	GL_CHECKED_CALL( glPopAttrib() );
}


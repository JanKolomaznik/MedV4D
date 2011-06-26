#include "AnnotationEditorController.hpp"


class IgnoreViewerController: public M4D::GUI::Viewer::AViewerController
{
public:
	typedef boost::shared_ptr< M4D::GUI::Viewer::AViewerController > Ptr;

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) { return false; }
};

class AnnotatePoints: public M4D::GUI::Viewer::AViewerController
{
public:
	typedef boost::shared_ptr< M4D::GUI::Viewer::AViewerController > Ptr;

	Qt::MouseButton	mVectorEditorInteractionButton;
	PointSet &mPoints;

	AnnotatePoints( PointSet &aPoints ): mVectorEditorInteractionButton( Qt::LeftButton ), mPoints( aPoints ) {}

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mVectorEditorInteractionButton && state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) {
			mPoints.addPoint( aEventInfo.realCoordinates );
			state.viewerWindow->update();
			return true;
		} 
		return false;
	}

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) { return false; }
};


AnnotationEditorController::AnnotationEditorController()
	: mEditMode( aemNONE )
{
	mVectorEditorInteractionButton = Qt::LeftButton;

	mOverlay = false;

		
	QActionGroup *group = new QActionGroup( this );
	group->setExclusive( false );

	QAction *action = new QAction( tr( "Point" ), this );
	action->setData( QVariant( aemPOINTS ) );
	action->setCheckable( true );
	group->addAction( action );
	mActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemPOINTS] = M4D::GUI::Viewer::AViewerController::Ptr( new AnnotatePoints(mPoints) );

	action = new QAction( tr( "Sphere" ), this );
	action->setData( QVariant( aemSPHERES ) );
	action->setCheckable( true );
	group->addAction( action );
	mActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemSPHERES] = M4D::GUI::Viewer::AViewerController::Ptr( new IgnoreViewerController );

	action = new QAction( tr( "Angle" ), this );
	action->setData( QVariant( aemANGLES ) );
	action->setCheckable( true );
	group->addAction( action );
	mActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemANGLES] = M4D::GUI::Viewer::AViewerController::Ptr( new IgnoreViewerController );

	action = new QAction( tr( "Line" ), this );
	action->setData( QVariant( aemLINES ) );
	action->setCheckable( true );
	group->addAction( action );
	mActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemLINES] = M4D::GUI::Viewer::AViewerController::Ptr( new IgnoreViewerController );

	QObject::connect( group, SIGNAL( triggered ( QAction * ) ), this, SLOT( editModeActionToggled( QAction * ) ) );
}

void
AnnotationEditorController::updateActions()
{
	for ( QList<QAction*>::iterator it = mActions.begin(); it != mActions.end(); ++it )
	{
		(*it)->setChecked( (*it)->data().value<int>() == mEditMode );
	}
}

/*bool
mouseMoveEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

bool	
mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );*/

bool
AnnotationEditorController::mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	//D_PRINT( "mousePressEvent" );

	if ( mEditMode != aemNONE ) {
		if ( mAnnotationPrimitiveHandlers[ mEditMode ]->mousePressEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	
	//D_PRINT( "mousePressEvent not handled" );
	return ControllerPredecessor::mousePressEvent ( aViewerState, aEventInfo );
}

/*bool
mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

bool
wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event );*/

unsigned
AnnotationEditorController::getAvailableViewTypes()const
{
	return M4D::GUI::Viewer::vt3D | M4D::GUI::Viewer::vt2DAlignedSlices;
}

void
AnnotationEditorController::render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane )
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
AnnotationEditorController::preRender3D()
{
	if( !mOverlay ) {
		render3D();
	}
}

void
AnnotationEditorController::postRender3D()
{
	if( mOverlay ) {
		render3D();
	}
}

void
AnnotationEditorController::render3D()
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


void
AnnotationEditorController::setAnnotationEditMode( int aMode )
{
	ASSERT( aMode < aemSENTINEL && aMode >= 0 ); //TODO

	//LOG( "setAnnotationEditMode - " << aMode );

	if( mEditMode != aemNONE ) {
		abortEditInProgress();

		emit updateRequest();
	}

	mEditMode = static_cast< AnnotationEditMode >( aMode );

	updateActions();	
}

void
AnnotationEditorController::abortEditInProgress()
{
	
}

QList<QAction *> &
AnnotationEditorController::getActions()
{
	return mActions;
}

void
AnnotationEditorController::editModeActionToggled( QAction *aAction )
{
	ASSERT( aAction != NULL );

	QVariant data = aAction->data();
	if ( data.canConvert<int>() ) {
		if ( aAction->isChecked() ) {
			setAnnotationEditMode( data.value<int>() );
		} else {
			setAnnotationEditMode( aemNONE );
		}
		return;
	}
	D_PRINT( "NOT HANDLED -----------------------" );
}

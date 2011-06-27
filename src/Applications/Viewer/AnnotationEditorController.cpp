#include "AnnotationEditorController.hpp"




class AnnotatePoints: public AnnotationBasicViewerController
{
public:
	typedef boost::shared_ptr< AnnotatePoints > Ptr;

	
	PointSet &mPoints;

	AnnotatePoints( PointSet &aPoints ): mPoints( aPoints ) {}


	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mVectorEditorInteractionButton && state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) {
			mPoints.push_back( M4D::Point3Df( aEventInfo.realCoordinates ) );
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

class AnnotateLines: public AnnotationBasicViewerController
{
public:
	typedef boost::shared_ptr< AnnotateLines > Ptr;
	
	LineSet &mLines;

	M4D::Line3Df *mCurrentLine;

	AnnotateLines( LineSet &aLines ): mLines( aLines ), mCurrentLine( NULL ) {}

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices 
			&& mCurrentLine != NULL	) 
		{
			mCurrentLine->secondPoint() = aEventInfo.realCoordinates;
			state.viewerWindow->update();
			return true;
		}
		return false; 
	}

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mVectorEditorInteractionButton 
			&& state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) 
		{
			if ( mCurrentLine == NULL ) {
				mLines.push_back( M4D::Line3Df( aEventInfo.realCoordinates, aEventInfo.realCoordinates ) );
				mCurrentLine = &(mLines[mLines.size()-1]);
			} else {
				mCurrentLine->secondPoint() = aEventInfo.realCoordinates;
				mCurrentLine = NULL;
			}
			state.viewerWindow->update();
			return true;
		} 
		if ( aEventInfo.event->button() == Qt::RightButton ) {
			abortEditation();
		}
		return false;
	}

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ return false; }

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) { return false; }

	void
	abortEditation()
	{
		if ( mCurrentLine != NULL ) {
			ASSERT( mLines.size() > 0 )
			//mLines.erase( mLines.end()-1, mLines.end() );
			mLines.resize( mLines.size()-1 );
		}
	}
};

class AnnotateSpheres: public AnnotationBasicViewerController
{
public:
	typedef boost::shared_ptr< AnnotateSpheres > Ptr;
	
	SphereSet &mSpheres;

	M4D::Sphere3Df *mCurrentSphere;

	AnnotateSpheres( SphereSet &aSpheres ): mSpheres( aSpheres ), mCurrentSphere( NULL ) {}

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices 
			&& mCurrentSphere != NULL	) 
		{
			mCurrentSphere->radius() = VectorSize(mCurrentSphere->center() - aEventInfo.realCoordinates);
			state.viewerWindow->update();
			return true;
		}
		return false; 
	}

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mVectorEditorInteractionButton 
			&& state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) 
		{
			if ( mCurrentSphere == NULL ) {
				mSpheres.push_back( M4D::Sphere3Df( aEventInfo.realCoordinates, 0.0f ) );
				mCurrentSphere = &(mSpheres[mSpheres.size()-1]);
			} else {
				mCurrentSphere->radius() = VectorSize(mCurrentSphere->center() - aEventInfo.realCoordinates);
				mCurrentSphere = NULL;
			}
			state.viewerWindow->update();
			return true;
		} 
		if ( aEventInfo.event->button() == Qt::RightButton ) {
			abortEditation();
		}
		return false;
	}

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ return false; }

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) { return false; }

	void
	abortEditation()
	{
		if ( mCurrentSphere != NULL ) {
			ASSERT( mSpheres.size() > 0 )
			//mSpheres.erase( mSpheres.end()-1, mSpheres.end() );
			mSpheres.resize( mSpheres.size()-1 );
		}
	}
};

//**************************************************************************
//**************************************************************************
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
	mAnnotationPrimitiveHandlers[aemPOINTS] = AnnotationBasicViewerController::Ptr( new AnnotatePoints(mPoints) );

	action = new QAction( tr( "Sphere" ), this );
	action->setData( QVariant( aemSPHERES ) );
	action->setCheckable( true );
	group->addAction( action );
	mActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemSPHERES] = AnnotationBasicViewerController::Ptr( new AnnotateSpheres( mSpheres ) );

	action = new QAction( tr( "Angle" ), this );
	action->setData( QVariant( aemANGLES ) );
	action->setCheckable( true );
	group->addAction( action );
	mActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemANGLES] = AnnotationBasicViewerController::Ptr( new AnnotationBasicViewerController );

	action = new QAction( tr( "Line" ), this );
	action->setData( QVariant( aemLINES ) );
	action->setCheckable( true );
	group->addAction( action );
	mActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemLINES] = AnnotationBasicViewerController::Ptr( new AnnotateLines( mLines ) );

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

bool
AnnotationEditorController::mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mEditMode != aemNONE ) {
		if ( mAnnotationPrimitiveHandlers[ mEditMode ]->mouseMoveEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseMoveEvent ( aViewerState, aEventInfo );
}
bool	
AnnotationEditorController::mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mEditMode != aemNONE ) {
		if ( mAnnotationPrimitiveHandlers[ mEditMode ]->mouseDoubleClickEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
		return true; //TODO - prevent plane changing
	}
	return ControllerPredecessor::mouseDoubleClickEvent ( aViewerState, aEventInfo );
}

bool
AnnotationEditorController::mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mEditMode != aemNONE ) {
		if ( mAnnotationPrimitiveHandlers[ mEditMode ]->mousePressEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mousePressEvent ( aViewerState, aEventInfo );
}

bool
AnnotationEditorController::mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mEditMode != aemNONE ) {
		if ( mAnnotationPrimitiveHandlers[ mEditMode ]->mouseReleaseEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseReleaseEvent ( aViewerState, aEventInfo );
}

bool
AnnotationEditorController::wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event )
{
	if ( mEditMode != aemNONE ) {
		if ( mAnnotationPrimitiveHandlers[ mEditMode ]->wheelEvent( aViewerState, event ) ) {
			return true;
		}
	}
	return ControllerPredecessor::wheelEvent ( aViewerState, event );
}

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
		for( size_t i = 0; i < mPoints.size(); ++i ) {
			if ( IntervalTest( aInterval[0], aInterval[1], mPoints[i][aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( mPoints[i], aPlane ) );
			}
		}
	glEnd();

	glBegin( GL_LINES );
		for( size_t i = 0; i < mLines.size(); ++i ) {
			//if ( IntervalTest( aInterval[0], aInterval[1], mPoints[i][aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( mLines[i].firstPoint(), aPlane ) );
				M4D::GLVertexVector( VectorPurgeDimension( mLines[i].secondPoint(), aPlane ) );
			//}
		}
	glEnd();
	glBegin( GL_POINTS );
		for( size_t i = 0; i < mLines.size(); ++i ) {
			if ( IntervalTest( aInterval[0], aInterval[1], mLines[i].firstPoint()[aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( mLines[i].firstPoint(), aPlane ) );
			}
			if ( IntervalTest( aInterval[0], aInterval[1], mLines[i].secondPoint()[aPlane] ) ) { 
				M4D::GLVertexVector( VectorPurgeDimension( mLines[i].secondPoint(), aPlane ) );
			}
		}
	glEnd();

	for( size_t i = 0; i < mSpheres.size(); ++i ) {
		Vector2f pos = VectorPurgeDimension( mSpheres[i].center(), aPlane );
		M4D::DrawCircle( pos, mSpheres[i].radius() );
	}

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

	
	GL_CHECKED_CALL( glDisable( GL_LIGHTING ) );
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );

	glBegin( GL_POINTS );
		for( size_t i = 0; i < mPoints.size(); ++i ) {
			M4D::GLVertexVector( mPoints[i] );
		}
		//std::for_each( mPoints.mPoints.begin(), mPoints.mPoints.end(), M4D::GLVertexVector );
	glEnd();

	glBegin( GL_LINES );
		for( size_t i = 0; i < mLines.size(); ++i ) {
			M4D::GLVertexVector( mLines[i].firstPoint() );
			M4D::GLVertexVector( mLines[i].secondPoint() );
		}
	glEnd();
	glBegin( GL_POINTS );
		for( size_t i = 0; i < mLines.size(); ++i ) {
			M4D::GLVertexVector( mLines[i].firstPoint() );
			M4D::GLVertexVector( mLines[i].secondPoint() );
		}
	glEnd();

	
	GL_CHECKED_CALL( glEnable( GL_LIGHTING ) );
	//GL_CHECKED_CALL( glEnable( GL_LIGHT0 ) );
	GL_CHECKED_CALL( glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, Vector4f( 1.0f, 0.0f, 0.0f, 1.0f ).GetData() ) );
	//GL_CHECKED_CALL( glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, Vector4f( 3.0f, 0.0f, 0.0f, 1.0f ).GetData() ) );
	for( size_t i = 0; i < mSpheres.size(); ++i ) {
		M4D::DrawSphere( mSpheres[i] );
	}

	GL_CHECKED_CALL( glPopAttrib() );
}


void
AnnotationEditorController::setAnnotationEditMode( int aMode )
{
	ASSERT( aMode < aemSENTINEL && aMode >= 0 ); //TODO

	//LOG( "setAnnotationEditMode - " << aMode );

	if( mEditMode != aemNONE ) {
		abortEditInProgress();
	}

	mEditMode = static_cast< AnnotationEditMode >( aMode );

	updateActions();	
}

void
AnnotationEditorController::abortEditInProgress()
{
	if ( mEditMode != aemNONE ) {
		mAnnotationPrimitiveHandlers[ mEditMode ]->abortEditation();	
		emit updateRequest();
	}
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

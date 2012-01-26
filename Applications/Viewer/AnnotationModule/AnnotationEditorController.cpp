#include "AnnotationModule/AnnotationEditorController.hpp"
#include "AnnotationModule/AnnotationSettingsDialog.hpp"

#include "AnnotationModule/AnnotationWidget.hpp"

#include "MedV4D/GUI/utils/OGLTools.h"


//**************************************************************************
//**************************************************************************
AnnotationEditorController::AnnotationEditorController()
	: mEditMode( aemNONE )
{
	mVectorEditorInteractionButton = Qt::LeftButton;

	mOverlay = false;

	//Chosen tool actions****************
	QActionGroup *group = new QActionGroup( this );
	group->setExclusive( false );

	QAction *action = new QAction( tr( "Point" ), this );
	action->setData( QVariant( aemPOINTS ) );
	action->setCheckable( true );
	group->addAction( action );
	mChosenToolActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemPOINTS] = APrimitiveCreationEventController::Ptr( new AnnotationPrimitiveController< M4D::Point3Df >(mPoints) );

	action = new QAction( tr( "Sphere" ), this );
	action->setData( QVariant( aemSPHERES ) );
	action->setCheckable( true );
	group->addAction( action );
	mChosenToolActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemSPHERES] = APrimitiveCreationEventController::Ptr( new AnnotationPrimitiveController< M4D::Sphere3Df >( mSpheres ) );

	action = new QAction( tr( "Angle" ), this );
	action->setData( QVariant( aemANGLES ) );
	action->setCheckable( true );
	group->addAction( action );
	mChosenToolActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemANGLES] = APrimitiveCreationEventController::Ptr(/* new AnnotationBasicViewerController */);

	action = new QAction( tr( "Line" ), this );
	action->setData( QVariant( aemLINES ) );
	action->setCheckable( true );
	group->addAction( action );
	mChosenToolActions.push_back( action );
	mAnnotationPrimitiveHandlers[aemLINES] = APrimitiveCreationEventController::Ptr( new AnnotationPrimitiveController<M4D::Line3Df>( mLines ) );

	QObject::connect( group, SIGNAL( triggered ( QAction * ) ), this, SLOT( editModeActionToggled( QAction * ) ) );

	mActions.append( mChosenToolActions );
	//****************************************
	
	action = new QAction( tr("Annotations Overlayed"), this ); //TODO better
	action->setCheckable( true );
	QObject::connect( action, SIGNAL( toggled ( bool ) ), this, SLOT( setOverlay(bool) ) );
	mActions.push_back( action );

	action = new QAction( tr("Annotation Settings"), this );
	QObject::connect( action, SIGNAL( triggered ( bool ) ), this, SLOT( showSettingsDialog() ) );
	mActions.push_back( action );

	mSettingsDialog = new AnnotationSettingsDialog();
	QObject::connect( mSettingsDialog, SIGNAL( applied() ), this, SLOT( applySettings() ) );

//*******************************
	mAnnotationView = new AnnotationWidget( &mPoints, &mSpheres, &mLines );
	QObject::connect( mAnnotationView, SIGNAL( annotationsCleared() ), this, SIGNAL( updateRequest() ) );
	

	//Settings
	mSettings.overlayed = false;

	mSettings.sphereFillColor2D = QColor( 255, 0, 0, 128 );
	mSettings.sphereContourColor2D = QColor( 255, 255, 255 );
	mSettings.sphereColor3D = QColor( 255, 0, 0 );
}

AnnotationEditorController::~AnnotationEditorController()
{
	delete mSettingsDialog;
}

QWidget *
AnnotationEditorController::getAnnotationView()
{
	return mAnnotationView;
}

void
AnnotationEditorController::updateActions()
{
	for ( QActionList::iterator it = mChosenToolActions.begin(); it != mChosenToolActions.end(); ++it )
	{
		(*it)->setChecked( (*it)->data().value<int>() == mEditMode );
	}
}

void
AnnotationEditorController::showSettingsDialog()
{
	ASSERT( mSettingsDialog != NULL );
	if ( mSettingsDialog->showDialog( mSettings ) ) {
		applySettings();
	}	
}

void
AnnotationEditorController::applySettings()
{
	mSettings = mSettingsDialog->getSettings();

	emit updateRequest();
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
	M4D::GLPushAtribs attribs;

	GL_CHECKED_CALL( glEnable( GL_POINT_SMOOTH ) );
	GL_CHECKED_CALL( glEnable( GL_BLEND ) );
	GL_CHECKED_CALL( glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) );
	GL_CHECKED_CALL( glPointSize( 4.0f ) );

	GL_CHECKED_CALL( glDisable( GL_DEPTH_TEST ) );
	
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );

	/*drawPointSet2D( mPoints.begin(), mPoints.end(), aInterval, aPlane );

	drawLineSet2D( mLines.begin(), mLines.end(), aInterval, aPlane );

	GL_CHECKED_CALL( M4D::GLColorFromQColor( mSettings.sphereFillColor2D ) );
	for( size_t i = 0; i < mSpheres.size(); ++i ) {
		float r2 = M4D::sqr( mSpheres[i].radius() ) - M4D::sqr( mSpheres[i].center()[ aPlane ] - 0.5*(aInterval[0]+aInterval[1]) );
		if ( r2 > 0.0f ) { 
			Vector2f pos = VectorPurgeDimension( mSpheres[i].center(), aPlane );
			M4D::drawCircle( pos, sqrt( r2 ) );
		}
	}*/
	
	renderPoints3D();

	renderLines3D();

	renderSpheres3D();
	
	renderAngles3D();
}

void
AnnotationEditorController::preRender3D()
{
	if( !mSettings.overlayed ) {
		render3D();
	}
}

void
AnnotationEditorController::postRender3D()
{
	if( mSettings.overlayed ) {
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

	//M4D::drawSphericalCap( Vector3f(), Vector3f( 0.0f, 0.0f, -1.0f ), 30.0f, 15.0f );
	
	renderPoints3D();

	renderLines3D();

	renderSpheres3D();
	
	renderAngles3D();

	GL_CHECKED_CALL( glPopAttrib() );
}

void
AnnotationEditorController::renderPoints2D()
{

}

void
AnnotationEditorController::renderSpheres2D()
{

}

void
AnnotationEditorController::renderLines2D()
{

}

void
AnnotationEditorController::renderAngles2D()
{

}
//------------------------------------------
void
AnnotationEditorController::renderPoints3D()
{
	GL_CHECKED_CALL( glDisable( GL_LIGHTING ) );
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );

	drawPointSet( mPoints.begin(), mPoints.end() );
}

void
AnnotationEditorController::renderSpheres3D()
{
	GL_CHECKED_CALL( glEnable( GL_LIGHTING ) );
	//GL_CHECKED_CALL( glEnable( GL_LIGHT0 ) );
	GL_CHECKED_CALL( glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, 
				Vector4f( 
					mSettings.sphereColor3D.redF(), 
					mSettings.sphereColor3D.greenF(), 
					mSettings.sphereColor3D.blueF(), 
					mSettings.sphereColor3D.alphaF() 
					).GetData() ) 
			);
	//GL_CHECKED_CALL( glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, Vector4f( 3.0f, 0.0f, 0.0f, 1.0f ).GetData() ) );
	for( size_t i = 0; i < mSpheres.size(); ++i ) {
		M4D::drawSphere( mSpheres[i] );
	}
}

void
AnnotationEditorController::renderLines3D()
{
	drawLineSet( mLines.begin(), mLines.end() );
}

void
AnnotationEditorController::renderAngles3D()
{

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
		mAnnotationPrimitiveHandlers[ mEditMode ]->cancel();	
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

	if( !(ApplicationManager::getInstance()->activateMode( mModeId )) ) {
		LOG( "Mode couldn't be activated" );
		return;
	}

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

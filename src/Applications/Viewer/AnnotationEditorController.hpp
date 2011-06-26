#ifndef ANNOTATION_EDITOR_CONTROLLER_HPP
#define ANNOTATION_EDITOR_CONTROLLER_HPP

#include <QtGui>

#include "GUI/widgets/GeneralViewer.h"
#include "GUI/utils/OGLTools.h"
#include "GUI/utils/OGLDrawing.h"
#include <algorithm>



//using namespace M4D;
class PointSet
{
public:
	void
	addPoint( Vector3f aCoord )
	{
		D_PRINT( "Adding point - " << aCoord );
		mPoints.push_back( aCoord );
		mSelectedIdx = mPoints.size() - 1;
		mSelected = true;
	}

	std::vector< Vector3f > mPoints;

	size_t mSelectedIdx;
	bool mSelected;
};

class AnnotationEditorController: public QObject, public M4D::GUI::Viewer::ViewerController, public M4D::GUI::Viewer::RenderingExtension
{
	Q_OBJECT;
public:
	enum AnnotationEditMode {
		aemNONE,
		aemPOINTS,
		aemSPHERES,
		aemLINES,
		aemANGLES,

		aemSENTINEL //use for valid interval testing 
	} 

	typedef boost::shared_ptr< AnnotationEditorController > Ptr;
	typedef M4D::GUI::Viewer::ViewerController ControllerPredecessor;
	typedef M4D::GUI::Viewer::RenderingExtension RenderingExtPredecessor;

	EditorController();

	/*bool
	mouseMoveEvent ( BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo );

	bool	
	mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo );*/

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo );

	/*bool
	mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo );

	bool
	wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event );*/

	unsigned
	getAvailableViewTypes()const;

	void
	render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane );

	void
	preRender3D();

	void
	postRender3D();

	void
	render3D();

public slots:
	void
	setAnnotationEditMode( int aMode );

	void
	abortEditInProgress();

signals:
	void
	updateRequest();
public:

	PointSet mPoints;
	Qt::MouseButton	mVectorEditorInteractionButton;

	bool mOverlay;
	AnnotationEditMode mEditMode;
};

#endif /*ANNOTATION_EDITOR_CONTROLLER_HPP*/

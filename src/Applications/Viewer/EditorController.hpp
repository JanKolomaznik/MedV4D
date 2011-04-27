#ifndef VIEWER_MODIFICATION_HPP
#define VIEWER_MODIFICATION_HPP

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

class EditorController: public M4D::GUI::Viewer::ViewerController, public M4D::GUI::Viewer::RenderingExtension
{
public:
	typedef boost::shared_ptr< EditorController > Ptr;
	typedef M4D::GUI::Viewer::ViewerController Predecessor1;

	EditorController();

	/*bool
	mouseMoveEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool	
	mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );*/

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	/*bool
	mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

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

	PointSet mPoints;
	Qt::MouseButton	mVectorEditorInteractionButton;

	bool mOverlay;
};

#endif /*VIEWER_MODIFICATION_HPP*/

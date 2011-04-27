#ifndef VIEWER_MODIFICATION_HPP
#define VIEWER_MODIFICATION_HPP

#include "GUI/widgets/GeneralViewer.h"
#include "GUI/utils/OGLDrawing.h"

class EditorController: public M4D::GUI::Viewer::ViewerController, public M4D::GUI::Viewer::RenderingExtension
{
public:
	typedef boost::shared_ptr< EditorController > Ptr;
	/*bool
	mouseMoveEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool	
	mouseDoubleClickEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool
	mousePressEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool
	mouseReleaseEvent ( BaseViewerState::Ptr aViewerState, QMouseEvent * event );

	bool
	wheelEvent ( BaseViewerState::Ptr aViewerState, QWheelEvent * event );*/

	unsigned
	getAvailableViewTypes()const
	{
		return M4D::GUI::Viewer::vt3D;
	}

	void
	render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane )
	{

	}

	void
	render3D()
	{
		/*glPushMatrix();
		glTranslatef( 10.0f, 10.0f, 10.0f );
		M4D::DrawSphere( 5 );

		glPopMatrix();*/
	}
};

#endif /*VIEWER_MODIFICATION_HPP*/

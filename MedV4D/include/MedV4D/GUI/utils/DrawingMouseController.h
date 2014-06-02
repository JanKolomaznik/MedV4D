#ifndef DRAWING_MOUSE_CONTROLLER_H
#define DRAWING_MOUSE_CONTROLLER_H

//Temporary workaround
#ifndef Q_MOC_RUN 

#include "MedV4D/GUI/widgets/AGLViewer.h"
#include "MedV4D/GUI/utils/MouseTracking.h"
#include "MedV4D/GUI/utils/ViewerController.h"
#endif
class ADrawingMouseController: public M4D::GUI::Viewer::AViewerController
{
	Q_OBJECT;
public:
	typedef std::shared_ptr< ADrawingMouseController > Ptr;
	
	Qt::MouseButton	mEditorInteractionButton;

	ADrawingMouseController(): mEditorInteractionButton( Qt::LeftButton ), mIsDrawing( false )
	{}
	
	~ADrawingMouseController()
	{}

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo );

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event );
signals:

protected:
	virtual void
	drawStep( const Vector3f &aStart, const Vector3f &aEnd ) = 0;
	
	M4D::GUI::Viewer::Mouse3DTrackInfo	mTrackInfo;
	bool mIsDrawing;
};


#endif /*DRAWING_MOUSE_CONTROLLER_H*/
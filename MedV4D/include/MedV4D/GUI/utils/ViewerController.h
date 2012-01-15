#ifndef VIEWER_CONTROLLER_H
#define VIEWER_CONTROLLER_H

#include "MedV4D/GUI/widgets/AGLViewer.h"
#include "MedV4D/GUI/utils/MouseTracking.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

class GeneralViewer;
	
class ViewerController: public AViewerController
{
	Q_OBJECT;
public:
	typedef boost::shared_ptr< ViewerController > Ptr;
	
	enum InteractionMode { 
		imNONE,
		imORBIT_CAMERA,
		imLUT_SETTING,
		imFAST_SLICE_CHANGE,
		imCUT_PLANE,
		imCUT_PLANE_OFFSET
	};

	ViewerController();

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo );

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo );

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo );

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo );

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event );

protected slots:
	virtual void
	timerCall();

protected:
	Qt::MouseButton	mCameraOrbitButton;
	Qt::MouseButton	mLUTSetMouseButton;
	Qt::MouseButton	mFastSliceChangeMouseButton;
	Qt::MouseButton	mCutPlaneOffsetButton;

	Qt::KeyboardModifiers mCutPlaneKeyboardModifiers;

	InteractionMode mInteractionMode;
	MouseTrackInfo	mTrackInfo;

	QTimer	mTimer;
	GeneralViewer *mTmpViewer;
	bool mPositive;
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*VIEWER_CONTROLLER_H*/
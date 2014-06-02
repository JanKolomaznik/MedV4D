#ifndef VIEWER_CONTROLLER_H
#define VIEWER_CONTROLLER_H

//Temporary workaround
#ifndef Q_MOC_RUN
#include "MedV4D/GUI/utils/AViewerController.h"
#include "MedV4D/GUI/utils/MouseTracking.h"
#endif

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
	typedef std::shared_ptr< ViewerController > Ptr;

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

#ifndef PROXY_VIEWER_CONTROLLER_H
#define PROXY_VIEWER_CONTROLLER_H

#include "GUI/widgets/GeneralViewer.h"

class ProxyViewerController: public M4D::GUI::Viewer::ViewerController
{
public:
	typedef boost::shared_ptr< ProxyViewerController > Ptr;

	ProxyViewerController()
	{}

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		if( mCurrentController ) {
			return mCurrentController->mouseMoveEvent ( aViewerState, aEventInfo );
		} 
		return M4D::GUI::Viewer::ViewerController::mouseMoveEvent ( aViewerState, aEventInfo ); 
	}

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		if( mCurrentController ) {
			return mCurrentController->mouseDoubleClickEvent ( aViewerState, aEventInfo );
		} 
		return M4D::GUI::Viewer::ViewerController::mouseDoubleClickEvent ( aViewerState, aEventInfo ); 
	}

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		if( mCurrentController ) {
			return mCurrentController->mousePressEvent ( aViewerState, aEventInfo );
		} 
		return M4D::GUI::Viewer::ViewerController::mousePressEvent ( aViewerState, aEventInfo ); 
	}

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		if( mCurrentController ) {
			return mCurrentController->mouseReleaseEvent ( aViewerState, aEventInfo );
		} 
		return M4D::GUI::Viewer::ViewerController::mouseReleaseEvent ( aViewerState, aEventInfo ); 
	}

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event )  
	{ 
		if( mCurrentController ) {
			return mCurrentController->wheelEvent ( aViewerState, event );
		} 
		return M4D::GUI::Viewer::ViewerController::wheelEvent ( aViewerState, event ); 
	}

	void
	setController( AViewerController::Ptr aCurrentController )
	{
		mCurrentController = aCurrentController;
	}

	AViewerController::Ptr
	getController()
	{
		return mCurrentController;
	}
protected:
	AViewerController::Ptr mCurrentController;
};

#endif /*PROXY_VIEWER_CONTROLLER_H*/

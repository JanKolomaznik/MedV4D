#ifndef PROXY_VIEWER_CONTROLLER_H
#define PROXY_VIEWER_CONTROLLER_H

#include "MedV4D/GUI/widgets/GeneralViewer.h"


class ModeViewerController: public M4D::GUI::Viewer::ViewerController
{
public:
	typedef boost::shared_ptr< ModeViewerController > Ptr;
	
	virtual void
	activated() = 0;

	virtual void
	deactivated() = 0;
};

template< typename TPredecessor >
class ProxyViewerControllerMixin: public TPredecessor
{
public:
	typedef boost::shared_ptr< ProxyViewerControllerMixin< TPredecessor > > Ptr;


	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		if( mCurrentController ) {
			return mCurrentController->mouseMoveEvent ( aViewerState, aEventInfo );
		} 
		return TPredecessor::mouseMoveEvent ( aViewerState, aEventInfo ); 
	}

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		if( mCurrentController ) {
			return mCurrentController->mouseDoubleClickEvent ( aViewerState, aEventInfo );
		} 
		return TPredecessor::mouseDoubleClickEvent ( aViewerState, aEventInfo ); 
	}

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		if( mCurrentController ) {
			return mCurrentController->mousePressEvent ( aViewerState, aEventInfo );
		} 
		return TPredecessor::mousePressEvent ( aViewerState, aEventInfo ); 
	}

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		if( mCurrentController ) {
			return mCurrentController->mouseReleaseEvent ( aViewerState, aEventInfo );
		} 
		return TPredecessor::mouseReleaseEvent ( aViewerState, aEventInfo ); 
	}

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event )  
	{ 
		if( mCurrentController ) {
			return mCurrentController->wheelEvent ( aViewerState, event );
		} 
		return TPredecessor::wheelEvent ( aViewerState, event ); 
	}

	void
	setController( M4D::GUI::Viewer::AViewerController::Ptr aCurrentController )
	{
		mCurrentController = aCurrentController;
	}

	M4D::GUI::Viewer::AViewerController::Ptr
	getController()
	{
		return mCurrentController;
	}
	
	void
	resetController()
	{
		mCurrentController.reset();
	}
protected:
	M4D::GUI::Viewer::AViewerController::Ptr mCurrentController;
};


typedef ProxyViewerControllerMixin< M4D::GUI::Viewer::ViewerController > ProxyViewerController;
typedef ProxyViewerControllerMixin< ModeViewerController > ModeProxyViewerController;


#endif /*PROXY_VIEWER_CONTROLLER_H*/

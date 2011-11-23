#ifndef MANAGER_VIEWER_SPECIAL_STATE
#define MANAGER_VIEWER_SPECIAL_STATE


#include "GUI/widgets/m4dGUISliceViewerWidget2.h"

template< typename ManagerType >
class ManagerViewerSpecialState: public M4D::Viewer::SliceViewerSpecialStateOperator
{
public:
	ManagerViewerSpecialState( ManagerType &manager ): _manager( manager )
	{  }

	void
	Draw( M4D::Viewer::SliceViewer & viewer, int sliceNum, double zoomRate )
	{
		_manager.Draw( sliceNum, zoomRate );
	}


	void 
	ButtonMethodRight( int amountH, int amountV, double zoomRate )
	{
		_manager.RightButtonMove( Vector< float32, 2 >( ((float32)amountH)/zoomRate, ((float32)amountV)/zoomRate ) );
	}
	
	void 
	ButtonMethodLeft( int amountH, int amountV, double zoomRate )
	{
		_manager.LeftButtonMove( Vector< float32, 2 >( ((float32)amountH)/zoomRate, ((float32)amountV)/zoomRate ) );
	}
	
	void 
	SelectMethodRight( double x, double y, int sliceNum, double zoomRate )
	{
		_manager.RightButtonDown( Vector< float32, 2 >( x, y ), sliceNum );
	}
	
	void 
	SelectMethodLeft( double x, double y, int sliceNum, double zoomRate )
	{
		_manager.LeftButtonDown( Vector< float32, 2 >( x, y ), sliceNum );
	}

	ManagerType	&_manager;

};
#endif /*MANAGER_VIEWER_SPECIAL_STATE*/

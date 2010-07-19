#ifndef A_3D_VIEWER_H
#define A_3D_VIEWER_H

#include "GUI/widgets/AGUIViewer.h"
#include <QtGui>
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace GUI
{
namespace Viewer
{

class I3DViewer
{
public:

	virtual void
	SetCameraPosition( const Vector<float32,3> &coordinates ) = 0;

	virtual void
	SetCameraTargetPosition( const Vector<float32,3> &coordinates ) = 0;

	virtual void
	SetCameraFieldOfView( float32 fov ) = 0;

	virtual void
	MoveCameraTarget( const Vector<float32,3> &translation ) = 0;

	virtual void
	MoveCamera( const Vector<float32,3> &translation ) = 0;

	virtual void
	SetCameraRotationXYZ( const Vector<float32,3> &angles ) = 0; 

	virtual void
	RotateCameraXYZ( const Vector<float32,3> &angles ) = 0; 

	virtual void
	RotateCameraAxis( float32 angle, const Vector<float32,3> &axis ) = 0;

	virtual void
	OrbitCamera( float32 angle, const Vector<float32,3> &axis ) = 0;


	/* TODO Light setup */



};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



#endif /*A_3D_VIEWER_H*/





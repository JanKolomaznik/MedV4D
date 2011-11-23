#ifndef A_2D_VIEWER_H
#define A_2D_VIEWER_H

#include "GUI/widgets/AGUIViewer.h"
#include <QtGui>
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace GUI
{
namespace Viewer
{

class I2DViewer
{
public:

	virtual void
	FlipVertical( bool flip ) = 0;

	virtual void
	FlipHorizontal( bool flip ) = 0;

	virtual void
	Flip( bool vertical, bool horizontal ) = 0;

	virtual void
	Zoom( float32 factor ) = 0;

	virtual void
	SetZoom( float32 factor ) = 0;

	virtual float32
	GetZoom() = 0;

	virtual void
	SetVerticalFieldOfView( float32 fov ) = 0;

	virtual float32
	GetVerticalFieldOfView()const = 0;
	
	virtual float32
	GetHorizontalFieldOfView()const = 0;

	virtual void
	SetCenterPosition( const Vector< float32, 2 > &coord ) = 0;

	virtual const Vector< float32, 2 > &
	GetCenterPosition( const Vector< float32, 2 > &coord )const = 0;
	
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



#endif /*A_2D_VIEWER_H*/





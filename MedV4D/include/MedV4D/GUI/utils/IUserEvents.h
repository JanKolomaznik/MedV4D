#ifndef INTERFACE_USER_EVENTS_H
#define INTERFACE_USER_EVENTS_H

#include <QtWidgets>
#include "MedV4D/Common/Common.h"

class IUserEvents
{
public:
	typedef std::shared_ptr< IUserEvents > Ptr;

	virtual
	~IUserEvents()
	{ }

	virtual bool
	mouseMoveEvent ( QSize aWinSize, QMouseEvent * event )
	{ return false; }

	virtual bool
	mouseDoubleClickEvent ( QSize aWinSize, QMouseEvent * event )
	{ return false; }

	virtual bool
	mousePressEvent ( QSize aWinSize, QMouseEvent * event )
	{ return false; }

	virtual bool
	mouseReleaseEvent ( QSize aWinSize, QMouseEvent * event )
	{ return false; }

	virtual bool
	wheelEvent ( QSize aWinSize, QWheelEvent * event )
	{ return false; }
};


#endif /*INTERFACE_USER_EVENTS_H*/

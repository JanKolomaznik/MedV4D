#ifndef INTERFACE_USER_EVENTS_H
#define INTERFACE_USER_EVENTS_H

#include <QtGui>

class IUserEvents
{
public:
	virtual bool
	mouseMoveEvent ( QMouseEvent * event ) = 0;

	virtual bool	
	mouseDoubleClickEvent ( QMouseEvent * event ) = 0;

	virtual bool
	mousePressEvent ( QMouseEvent * event ) = 0;

	virtual bool
	mouseReleaseEvent ( QMouseEvent * event ) = 0;

	virtual bool
	wheelEvent ( QWheelEvent * event ) = 0;
};


#endif /*INTERFACE_USER_EVENTS_H*/

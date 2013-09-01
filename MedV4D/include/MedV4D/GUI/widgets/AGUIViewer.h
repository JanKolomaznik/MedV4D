#ifndef A_GUI_VIEWER_H
#define A_GUI_VIEWER_H

#include "MedV4D/GUI/widgets/AViewer.h"
#include <QtWidgets>
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace GUI
{
namespace Viewer
{

/*class AGUIViewer;

enum{
	EID_KEY_PRESS,
	EID_KEY_RELEASE,
	EID_MOUSE_DOUBLE_CLICK,
	EID_MOUSE_PRESS,
	EID_MOUSE_RELEASE,
	EID_MOUSE_MOVE,
	EID_MOUSE_WHEEL,

	EID_LIMIT
};

class AEventHandler
{
public:
	typedef boost::shared_ptr< AEventHandler >	Ptr;

	AEventHandler()
	{ for( unsigned i = 0; i < EID_LIMIT; ++i ) _handleFlags[i] = false; }

	bool
	IsHandlingEvent( uint32 eventID )const
	{ return _handleFlags[ eventID ]; }

	virtual void
	keyPressEvent ( AGUIViewer &viewer, QKeyEvent * event ) = 0;

	virtual void
	keyReleaseEvent ( AGUIViewer &viewer, QKeyEvent * event ) = 0;

	virtual void
	mouseDoubleClickEvent ( AGUIViewer &viewer, QMouseEvent * event ) = 0;

	virtual void
	mouseMoveEvent ( AGUIViewer &viewer, QMouseEvent * event ) = 0;

	virtual void
	mousePressEvent ( AGUIViewer &viewer, QMouseEvent * event ) = 0;

	virtual void
	mouseReleaseEvent ( AGUIViewer &viewer, QMouseEvent * event ) = 0;

	virtual void
	wheelEvent ( AGUIViewer &viewer, QWheelEvent * event ) = 0;

protected:
	bool	_handleFlags[ EID_LIMIT ];
};*/


class AGUIViewer: public AViewer
{
public:


	/**
	* Cast explicitly the viewer to a QWidget. It is necessary for being able to add
	* the widget to other Qt widgets - this class has only a QObject base; the inheriting
	* class has to inherit from QWidget (the reason for this is the problem of multiple
	* inheritence, since the inheriting class will probably inherit from another subclass
	* of QWidget, like QVTKWidget or QGLWidget).
	*  @return Pointer that is casted to the QWidget base of the implementing class
	**/
	virtual QWidget*
	CastToQWidget()=0;

	/*void
	SetEventHandler( AEventHandler::Ptr handler )
	{ _eventHandler = handler; }*/


protected:

	//AEventHandler::Ptr	_eventHandler;

private:

};


#define USER_QTEVENT_BINDING_MACRO \
protected:\
void keyPressEvent ( QKeyEvent * event ) \
{\
	if( _eventHandler && _eventHandler->IsHandlingEvent( EID_KEY_PRESS ) ) { \
		_eventHandler->keyPressEvent( *this, event );\
	} else { event->ignore(); }\
}\
\
void keyReleaseEvent ( QKeyEvent * event )\
{\
	if( _eventHandler && _eventHandler->IsHandlingEvent( EID_KEY_RELEASE ) ) { \
		_eventHandler->keyReleaseEvent( *this, event );\
	} else { event->ignore(); }\
}\
\
void mouseDoubleClickEvent ( QMouseEvent * event )\
{\
	if( _eventHandler && _eventHandler->IsHandlingEvent( EID_MOUSE_DOUBLE_CLICK ) ) { \
		_eventHandler->mouseDoubleClickEvent( *this, event );\
	} else { event->ignore(); }\
}\
\
void mouseMoveEvent ( QMouseEvent * event )\
{\
	if( _eventHandler && _eventHandler->IsHandlingEvent( EID_MOUSE_MOVE ) ) { \
		_eventHandler->mouseMoveEvent( *this, event );\
	} else { event->ignore(); }\
}\
\
void mousePressEvent ( QMouseEvent * event )\
{\
	if( _eventHandler && _eventHandler->IsHandlingEvent( EID_MOUSE_PRESS ) ) { \
		_eventHandler->mousePressEvent( *this, event );\
	} else { event->ignore(); }\
}\
\
void mouseReleaseEvent ( QMouseEvent * event )\
{\
	if( _eventHandler && _eventHandler->IsHandlingEvent( EID_MOUSE_RELEASE ) ) { \
		_eventHandler->mouseReleaseEvent( *this, event );\
	} else { event->ignore(); }\
}\
\
void wheelEvent ( QWheelEvent * event )\
{\
	if( _eventHandler && _eventHandler->IsHandlingEvent( EID_MOUSE_WHEEL ) ) { \
		_eventHandler->wheelEvent( *this, event );\
	} else { event->ignore(); }\
}


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



#endif /*A_GUI_VIEWER_H*/


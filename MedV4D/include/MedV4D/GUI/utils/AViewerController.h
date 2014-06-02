#ifndef AVIEWER_CONTROLLER_H
#define AVIEWER_CONTROLLER_H

#include <QtCore>
#include <QColor>
#include <QWheelEvent>
#include <memory>
#include <boost/cast.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/ext.hpp>

#include <soglu/GLViewSetup.hpp>
#include <soglu/GLSLShader.hpp>

namespace M4D
{
namespace GUI
{
namespace Viewer
{

typedef glm::fvec3 Point3Df;
struct MouseEventInfo;

enum ViewType
{
	vt2DAlignedSlices	= 1,
	vt2DGeneralSlices	= 1 << 1,
	vt3D			= 1 << 2
};




class BaseViewerState
{
public:
	typedef std::shared_ptr< BaseViewerState > Ptr;
	virtual ~BaseViewerState(){}

	glm::uvec2	mWindowSize;
	float		aspectRatio;

	QWidget		*viewerWindow;

	QColor		backgroundColor;

	unsigned availableViewTypes;
	ViewType viewType;

	soglu::GLViewSetup 	glViewSetup;

	template< typename TViewerType >
	TViewerType &
	getViewerWindow()
	{
		return *boost::polymorphic_cast< TViewerType *>( viewerWindow );//TODO exceptions
	}
	soglu::GLSLProgram basicShaderProgram;
};

class AViewerController: public QObject
{
public:
	typedef std::shared_ptr< AViewerController > Ptr;

	virtual bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo ) = 0;

	virtual bool
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo ) = 0;

	virtual bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo ) = 0;

	virtual bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const MouseEventInfo &aEventInfo ) = 0;

	virtual bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) = 0;
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

#endif /*VIEWER_CONTROLLER_H*/


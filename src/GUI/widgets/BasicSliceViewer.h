#ifndef BASIC_SLICE_VIEWER_H
#define BASIC_SLICE_VIEWER_H

#include "GUI/utils/CgShaderTools.h"
#include "GUI/utils/GLTextureImage.h"
#include "GUI/widgets/AGUIViewer.h"
#include "GUI/widgets/ViewerConstructionKit.h"
#include <QtGui>
#include <QtOpenGL>
#include <boost/shared_ptr.hpp>
#include "Imaging/Imaging.h"
#include "GUI/utils/ViewConfiguration.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

class BasicSliceViewer : 
	public ViewerConstructionKit<   QGLWidget, 
					PortInterfaceHelper< boost::mpl::vector< M4D::Imaging::AImage > >
					>
{
	Q_OBJECT;
public:
	typedef ViewerConstructionKit<   QGLWidget, 
					PortInterfaceHelper< boost::mpl::vector< M4D::Imaging::AImage > >
					>	PredecessorType;
	

	BasicSliceViewer( QWidget *parent = NULL );

	~BasicSliceViewer();

	void
	SetLUTWindow( float32 center, float32 width );

	void
	SetLUTWindow( Vector< float32, 2 > window );

	const Vector< float32, 2 > &
	GetLUTWindow()
		{ return _lutWindow; }

	void
	SetCurrentSlice( int32 slice );

	void
	ZoomFit( ZoomType zoomType = ztFIT );

	/*void
	SetImage( M4D::Imaging::AImage::Ptr image )
	{ _image = image; }*/

protected:
	void	
	initializeGL ();

	void	
	initializeOverlayGL ();

	void	
	paintGL ();

	void	
	paintOverlayGL ();

	void	
	resizeGL ( int width, int height );

	void
	mouseMoveEvent ( QMouseEvent * event );

	void	
	resizeOverlayGL ( int width, int height );

	void
	mousePressEvent ( QMouseEvent * event );

	void
	mouseReleaseEvent ( QMouseEvent * event );

	void
	wheelEvent ( QWheelEvent * event );


	bool
	IsDataPrepared();

	bool
	PrepareData();

	void
	RenderOneDataset();

	enum {rmONE_DATASET}	_renderingMode;

	CartesianPlanes		_plane;

	int32			_currentSlice;

	GLTextureImage::Ptr	_textureData;

	ViewConfiguration2D	_viewConfiguration;

	CGcontext   				_cgContext;
	CgBrightnessContrastShaderConfig	_shaderConfig;

	Vector< float, 3 > 			_regionRealMin;
	Vector< float, 3 >			_regionRealMax;
	Vector< float, 3 >			_elementExtents;
	Vector< int32, 3 > 			_regionMin;
	Vector< int32, 3 >			_regionMax;

	QPoint					_clickPosition;

	Vector< float32, 2 > 			_lutWindow;
	Vector< float32, 2 > 			_oldLUTWindow;

	 enum InteractionMode { 
		 imNONE,
		 imSETTING_LUT_WINDOW 
	 }					_interactionMode;
	 bool					_prepared;
	//M4D::Imaging::AImage::Ptr 		_image;

private:

};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/



#endif /*BASIC_SLICE_VIEWER_H*/




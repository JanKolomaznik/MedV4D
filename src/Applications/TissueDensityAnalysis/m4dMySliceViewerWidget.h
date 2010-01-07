/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUISliceViewerWidget2.h 
 * @{ 
 **/

#ifndef _M4DMYSLICEVIEWERWIDGET_H
#define _M4DMYSLICEVIEWERWIDGET_H

#include "GUI/widgets/m4dGUISliceViewerWidget.h"
//#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"

#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace Viewer
{

class m4dMySliceViewerWidget;

typedef m4dMySliceViewerWidget SliceViewer;

/**
 * Sliceviewer's texture preparer that shows the first input dataset as a greyscale image
 */
template< typename ElementType >
class MySimpleSliceViewerTexturePreparer : public SimpleSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    MySimpleSliceViewerTexturePreparer(): SimpleSliceViewerTexturePreparer< ElementType >() {}

    /**
     * Prepares the texture of the image to be mapped to the following OpenGL surface.
     *  @param inputPorts the input pipeline port list to get the image from
     *  @param width reference to set the width of the texture
     *  @param height reference to set the height of the texture
     *  @param brightnessRate the rate of brightness to adjust the image with
     *  @param contrastRate the rate of contrast to adjust the image with
     *  @param so the orientation of the slices (xy, yz, zx)
     *  @param slice the number of the slice to be drawn
     *  @param dimension dataset's number of dimensions
     *  @return true, if texture preparing was successful, false otherwise
     */
    virtual bool prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension );

protected:

    /**
     * Function that arranges the voxels in correct order.
     *  @param dst pointer to the destination array
     *  @param src pointer to the source array
     *  @param width the width of the image
     *  @param height the height of the image
     *  @param newWidth the new width of the image after texture correction ( to be a power of 2 )
     *  @param newHeight the new height of the image after texture correction ( to be a power of 2 )
     *  @param depth the depth at which the slice lies
     *  @param xstride the steps between two neighbor voxels according to coordinate x
     *  @param ystride the steps between two neighbor voxels according to coordinate y
     *  @param zstride the steps between two neighbor voxels according to coordinate z
     */
    void copy( ElementType* dst, ElementType* src, uint32 width, uint32 height, uint32 newWidth, uint32 newHeight, uint32 depth, int32 xstride, int32 ystride, int32 zstride )
    {
        uint32 i, j;
        for ( i = 0; i < newHeight; i++ )
            for ( j = 0; j < newWidth; j++ )
		if ( i < height && j < width ) dst[ i * newWidth + j ] = src[ j * xstride + i * ystride + depth * zstride ];
		else dst[ i * newWidth + j ] = 0;
    }

	/**
     * Function that arranges the voxels in correct order.
     *  @param dst pointer to the destination array
     *  @param src pointer to the source array
     *  @param width the width of the image
     *  @param height the height of the image
     *  @param newWidth the new width of the image after texture correction ( to be a power of 2 )
     *  @param newHeight the new height of the image after texture correction ( to be a power of 2 )
     *  @param depth the depth at which the slice lies
     *  @param xstride the steps between two neighbor voxels according to coordinate x
     *  @param ystride the steps between two neighbor voxels according to coordinate y
     *  @param zstride the steps between two neighbor voxels according to coordinate z
     */
    void maskCopy( ElementType* dst, ElementType* src, ElementType* mask, uint32 width, uint32 height, uint32 newWidth, uint32 newHeight, uint32 depth, int32 xstride, int32 ystride, int32 zstride )
    {
        uint32 i, j;
        for ( i = 0; i < newHeight; i++ )
            for ( j = 0; j < newWidth; j++ )
				if ( i < height && j < width && mask[ j * xstride + i * ystride + depth * zstride ] > 0) 
					dst[ i * newWidth + j ] = src[ j * xstride + i * ystride + depth * zstride ];
				else 
					dst[ i * newWidth + j ] = 0;
    }

	/**
     * Prepares the texture of the image to be mapped to the following OpenGL surface.
     *  @param inPort the input port to get the image from
     *  @param width reference to set the width of the texture
     *  @param height reference to set the height of the texture
     *  @param so the orientation of the slices (xy, yz, zx)
     *  @param slice the number of the slice to be drawn
     *  @param dimension dataset's number of dimensions
     *  @return pointer to the resulting texture array, if texture preparing was successful, NULL otherwise
     */
    ElementType* prepareSingle( Imaging::InputPortTyped<Imaging::AImage>* inPort,
				Imaging::InputPortTyped<Imaging::AImage>* inMaskPort,
      uint32& width,
      uint32& height,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension );

    /**
     * Prepares several texture arrays of datasets
     *  @param inputPorts the input pipeline port list to get the image from
     *  @param numberOfDatasets the number of datasets to be arranged and returned
     *  @param width reference to set the width of the texture
     *  @param height reference to set the height of the texture
     *  @param so the orientation of the slices (xy, yz, zx)
     *  @param slice the number of the slice to be drawn
     *  @param dimension dataset's number of dimensions
     *  @return array of arrays of the prepared textures
     */
    virtual ElementType** getDatasetArrays( const Imaging::InputPortList& inputPorts,
      uint32 numberOfDatasets,
      uint32& width,
      uint32& height,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension );
};

class SliceViewerSpecialStateOperator
{
public:
	SliceViewerSpecialStateOperator(){};
/*
	 void
	Draw( SliceViewer & viewer, int sliceNum, double zoomRate );

	 void 
	ButtonMethodRight( int amountH, int amountV, double zoomRate );
	
	 void 
	ButtonMethodLeft( int amountH, int amountV, double zoomRate );
	
	 void 
	SelectMethodRight( double x, double y, int sliceNum, double zoomRate );
	*/
	 void 
	SelectMethodLeft( double x, double y, int sliceNum, double zoomRate );
};

typedef boost::shared_ptr< SliceViewerSpecialStateOperator >	SliceViewerSpecialStateOperatorPtr;


class m4dMySliceViewerWidget: public m4dGUISliceViewerWidget, QObject
{
    Q_OBJECT

public:
	typedef m4dGUISliceViewerWidget	PredecessorType;

    /**
     * Constructor.
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    m4dMySliceViewerWidget( unsigned index, QWidget *parent = 0 );

    /**
     * Construtor.
     *  @param conn the connection that connects the viewer
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    m4dMySliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 );
	
	void setMaskConnection(Imaging::ConnectionInterface* connMask);

	//void drawSliceAdditionals( int sliceNum, double zoomRate );

	void drawHUD( int sliceNum, double zoomRate, QPoint offset );

	void setSpecialState( SliceViewerSpecialStateOperatorPtr state )
		{ _specialState = state; }

	void setButtonHandler( ButtonHandler hnd, MouseButton btn );
	
	void makeConnections();

public slots:

	void slotSetSpecialStateSelectMethodLeft();
	
	/**
     * Starts selecting a new shape in the image.
     *  @param x the x coordinate of the first point of the shape
     *  @param y the y coordinate of the first point of the shape
     *  @param z the z coordinate of the first point of the shape
     */
    void sphereCenter( double x, double y, double z );


signals:
	
	void signalSphereCenter(double x, double y, double z );
	
protected: 
  Imaging::InputPortTyped<Imaging::AImage>* _inMaskPort;

	void specialStateButtonMethodLeft( int amountA, int amountB );
/*
	void specialStateButtonMethodRight( int amountA, int amountB );
*/
	void specialStateSelectMethodLeft( double x, double y, double z );
/*
	void specialStateSelectMethodRight( double x, double y, double z );

	*/
	
	//virtual void mousePressEvent(QMouseEvent *event);

	SliceViewerSpecialStateOperatorPtr		_specialState;

    /**
     * Draws a slice.
     *  @param sliceNum the number of the slice that is to be drawn
     *  @param zoomRate the zoom rate that is to be applied to the image
     *  @param offset the offset of the image on the viewer
     */
    void drawSlice( int sliceNum, double zoomRate, QPoint offset );
	
	
};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

/** @} */


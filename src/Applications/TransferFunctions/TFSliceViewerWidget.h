#ifndef TF_SLICEVIEWERWIDGET
#define TF_SLICEVIEWERWIDGET

#include "GUI/widgets/m4dGUISliceViewerWidget.h"
#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"
#include "GUI/widgets/utils/ViewerFactory.h"

#include <boost/shared_ptr.hpp>

#include <TFSimpleFunction.h>

namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer that shows the first input dataset as a greyscale image
 */
template< typename ElementType >
class TFSimpleSliceViewerTexturePreparer : public SimpleSliceViewerTexturePreparer< ElementType >
{

public:

	TFSimpleSliceViewerTexturePreparer():
		SimpleSliceViewerTexturePreparer< ElementType >(),
		tfUsed_(false),
		histSlice_(-1){

		currentTransferFunction_ = std::vector<ElementType>(TypeTraits<ElementType>::Max);
	}

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
    virtual bool prepare(
		const Imaging::InputPortList& inputPorts,
		uint32& width,
		uint32& height,
		GLint brightnessRate,
		GLint contrastRate,
		SliceOrientation so,
		uint32 slice,
		unsigned& dimension );

	void setTransferFunction(TFAbstractFunction &transferFunction);

	std::vector<int> getHistogram();

private:
	std::vector<ElementType> currentTransferFunction_;
	bool tfUsed_;
	std::vector<unsigned> histogram_;
	uint32 histSlice_;
};

class TFSliceViewerWidget: public m4dGUISliceViewerWidget, public QObject{

	Q_OBJECT

public:
	
	typedef m4dGUISliceViewerWidget	PredecessorType;

    /**
     * Constructor.
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    TFSliceViewerWidget( unsigned index, QWidget *parent = 0 ):
		PredecessorType( index, parent ), currentImageID_(-1), texturePreparer_(NULL){}

    /**
     * Construtor.
     *  @param conn the connection that connects the viewer
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    TFSliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 ):
		PredecessorType( conn, index, parent ), currentImageID_(-1), texturePreparer_(NULL){}

	~TFSliceViewerWidget(){
		delete texturePreparer_;
	}
/*
signals:
	void Histogram(std::vector<int> hist);
*/
public slots:
	void adjust_by_transfer_function(TFAbstractFunction &transferFunction);
	//void send_histogram();
	
protected:   
	int currentImageID_;
	AbstractSliceViewerTexturePreparer* texturePreparer_;

    /**
     * Draws a slice.
     *  @param sliceNum the number of the slice that is to be drawn
     *  @param zoomRate the zoom rate that is to be applied to the image
     *  @param offset the offset of the image on the viewer
     */
    //void drawSlice( int sliceNum, double zoomRate, QPoint offset );	
};

typedef M4D::GUI::GenericViewerFactory< M4D::Viewer::TFSliceViewerWidget > TFViewerFactory;

} /*namespace Viewer*/
} /*namespace M4D*/


#endif	//TF_SLICEVIEWERWIDGET
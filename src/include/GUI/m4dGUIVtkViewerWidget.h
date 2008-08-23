#ifndef M4D_GUI_VTK_VIEWER_WIDGET_H
#define M4D_GUI_VTK_VIEWER_WIDGET_H

#include <QWidget>
#include "QVTKWidget.h"

// VTK includes
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkVolume.h"
#include "vtkActor2D.h"
#include "vtkPolyDataMapper2D.h"
#include "vtkProperty2D.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkCamera.h"
#include "vtkVolumeRayCastMapper.h"
#include "vtkVolumeRayCastCompositeFunction.h"
#include "vtkImageCast.h"
#include "vtkPiecewiseFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkVolumeProperty.h"

#include "vtkIntegration/m4dImageDataSource.h"

#include "Common.h"
#include "Imaging/Image.h"
#include "Imaging/Ports.h"
#include "Imaging/ImageConnection.h"
#include "GUI/m4dGUIAbstractViewerWidget.h"

namespace M4D
{
namespace Viewer
{

class m4dGUIVtkViewerWidget: public m4dGUIAbstractViewerWidget, public QVTKWidget
{
    Q_OBJECT

public:
    m4dGUIVtkViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 );
    m4dGUIVtkViewerWidget( unsigned index, QWidget *parent = 0 );
    ~m4dGUIVtkViewerWidget();

    virtual void setInputPort();
    virtual void setInputPort( Imaging::ConnectionInterface* conn );

    virtual AvailableSlots getAvailableSlots();

    virtual QWidget* operator()();

    virtual void ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction );

public slots:

    /**
     * Slot to connect a given button to a given handler method
     * (it has no function in this type of viewer)
     *  @param hnd the handler method
     *  @param btn the button to connect to the method
     */
    virtual void slotSetButtonHandler( ButtonHandler hnd, MouseButton btn );
    
    /**
     * Slot to set if the viewer is selected or not
     *  @param selected tells if the viewer should be selected or not
     */
    virtual void slotSetSelected( bool selected );

    /**
     * Slot to set the current slice number
     * (it has no function in this type of viewer)
     *  @param num the slice number to be set
     */
    virtual void slotSetSliceNum( size_t num );

    /**
     * Slot to set the viewer to show one slice at once
     * (it has no function in this type of viewer)
     */
    virtual void slotSetOneSliceMode();

    /**
     * Slot to set the viewer to show several slices at once
     * (it has no function in this type of viewer)
     *  @param slicesPerRow how many slices will be shown in one row
     */
    virtual void slotSetMoreSliceMode( unsigned slicesPerRow );

    /**
     * Slot to toggle vertical flip
     * (it has no function in this type of viewer)
     */
    virtual void slotToggleFlipVertical();

    /**
     * Slot to toggle horizontal flip
     * (it has no function in this type of viewer)
     */
    virtual void slotToggleFlipHorizontal();

    /**
     * Slot to add some text data to show on the left side of the viewer
     * (it has no function in this type of viewer)
     *  @param type the type of the given data
     *  @param data the value of the given data
     */
    virtual void slotAddLeftSideData( std::string type, std::string data );

    /**
     * Slot to add some text data to show on the right side of the viewer
     * (it has no function in this type of viewer)
     *  @param type the type of the given data
     *  @param data the value of the given data
     */
    virtual void slotAddRightSideData( std::string type, std::string data );

    /**
     * Slot to erase some data from the left side of the viewer
     * (it has no function in this type of viewer)
     *  @param type the type of the data that is to be erased
     */
    virtual void slotEraseLeftSideData( std::string type );

    /**
     * Slot to erase some data from the right side of the viewer
     * (it has no function in this type of viewer)
     *  @param type the type of the data that is to be erased
     */
    virtual void slotEraseRightSideData( std::string type );

    /**
     * Slot to clear all data from the left side of the viewer
     * (it has no function in this type of viewer)
     */
    virtual void slotClearLeftSideData();

    /**
     * Slot to clear all data from the right side of the viewer
     * (it has no function in this type of viewer)
     */
    virtual void slotClearRightSideData();

    /**
     * Slot to toggle the printing of data on the viewer
     * (it has no function in this type of viewer)
     */
    virtual void slotTogglePrintData();

    /**
     * Slot to toggle the printing of the selected shapes' information on the viewer
     * (it has no function in this type of viewer)
     */
    virtual void slotTogglePrintShapeData();
    
    /**
     * Slot to zoom the image
     *  @param amount how much we want to zoom. Positive value means zoom in,
     *                negative value means zoom out.
     */
    virtual void slotZoom( int amount );

    /**
     * Slot to move the image
     * (it has no function in this type of viewer)
     *  @param amountH the amount to move the image horizontally
     *  @param amountV the amount to move the image vertically
     */
    virtual void slotMove( int amountH, int amountV );

    /**
     * Slot to adjust the brightness and contrast of the image
     * (it has no function in this type of viewer)
     *  @param amountB the amount to adjust the brightness
     *  @param amountC the amount to adjust the contrast
     */
    virtual void slotAdjustContrastBrightness( int amountB, int amountC );

    /**
     * Slot to add a new point to the last created shape of the list of selected
     * shapes
     * (it has no function in this type of viewer)
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     *  @param z the z coordinate of the point
     */
    virtual void slotNewPoint( double x, double y, double z );

    /**
     * Slot to add a new shape to the list of selected shapes
     * (it has no function in this type of viewer)
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     *  @param z the z coordinate of the point
     */
    virtual void slotNewShape( double x, double y, double z );

    /**
     * Slot to delete the last selected point
     * (it has no function in this type of viewer)
     */
    virtual void slotDeletePoint();
    
    /**
     * Slot to delete the last selected shape
     * (it has no function in this type of viewer)
     */
    virtual void slotDeleteShape();

    /**
     * Slot to erase all selected shapes and poitns
     * (it has no function in this type of viewer)
     */
    virtual void slotDeleteAll();

    /**
     * Slot to rotate the scene around the x axis
     *  @param x the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisX( double x );

    /**
     * Slot to rotate the scene around the y axis
     *  @param y the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisY( double y );

    /**
     * Slot to rotate the scene around the z axis
     *  @param z the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisZ( double z );

    /**
     * Slot to toggle the orientation of the slice viewing axes
     * xy -> yz -> zx
     * (it has no function in this type of viewer)
     */
    virtual void slotToggleSliceOrientation();

    /**
     * Slot to pick the color of the pixel at the given position
     * (it has no function in this type of viewer)
     *  @param x the x coordinate
     *  @param y the y coordinate
     *  @param z the z coordinate
     */
    virtual void slotColorPicker( double x, double y, double z );

protected slots:
    
    /**
     * Slot to handle incoming message from Image pipeline
     *  @param msgID the ID of the message
     */
    virtual void slotMessageHandler( Imaging::PipelineMsgID msgID );

protected:
    virtual void resizeEvent( QResizeEvent* event );
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void setUnSelected();
    virtual void setSelected();

private:
    void setBorderPoints( vtkPoints* points, vtkCellArray *cells, unsigned pos );
    void setParameters();

    bool					_plugged;

    Imaging::InputPortAbstractImage*		_inPort;
    vtkIntegration::m4dImageDataSource*		_imageData;
    vtkImageCast*				_iCast;
    vtkPiecewiseFunction*			_opacityTransferFunction;
    vtkColorTransferFunction*			_colorTransferFunction;
    vtkVolumeProperty*				_volumeProperty;
    vtkVolumeRayCastMapper*			_volumeMapper;
    vtkVolume*					_volume;
    vtkActor2D*					_actor2DSelected;
    vtkPoints*					_pointsSelected;
    vtkPolyData*				_pointsDataSelected;
    vtkPolyDataMapper2D*			_pointsDataMapperSelected;
    vtkCellArray*				_cellsSelected;
    vtkActor2D*					_actor2DPlugged;
    vtkPoints*					_pointsPlugged;
    vtkPolyData*				_pointsDataPlugged;
    vtkPolyDataMapper2D*			_pointsDataMapperPlugged;
    vtkCellArray*				_cellsPlugged;
    vtkRenderer*				_renImageData;
    AvailableSlots				_availableSlots;
};

} /* namespace Viewer */
} /* namespace M4D */

#endif // M4D_GUI_VTK_VIEWER_WIDGET_H

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

#include "m4dImageDataSource.h"

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
    virtual void slotSetButtonHandler( ButtonHandler hnd, MouseButton btn );
    virtual void slotSetSelected( bool selected );
    virtual void slotSetSliceNum( size_t num );
    virtual void slotSetOneSliceMode();
    virtual void slotSetMoreSliceMode( unsigned slicesPerRow );
    virtual void slotToggleFlipVertical();
    virtual void slotToggleFlipHorizontal();
    virtual void slotAddLeftSideData( std::string type, std::string data );
    virtual void slotAddRightSideData( std::string type, std::string data );
    virtual void slotEraseLeftSideData( std::string type );
    virtual void slotEraseRightSideData( std::string type );
    virtual void slotClearLeftSideData();
    virtual void slotClearRightSideData();
    virtual void slotTogglePrintData();
    virtual void slotTogglePrintShapeData();
    virtual void slotZoom( int amount );
    virtual void slotMove( int amountH, int amountV );
    virtual void slotAdjustContrastBrightness( int amountB, int amountC );
    virtual void slotNewPoint( int x, int y, int z );
    virtual void slotNewShape( int x, int y, int z );
    virtual void slotDeletePoint();
    virtual void slotDeleteShape();
    virtual void slotDeleteAll();
    virtual void slotRotateAxisX( double x );
    virtual void slotRotateAxisY( double y );
    virtual void slotRotateAxisZ( double z );

protected:
    virtual void resizeEvent( QResizeEvent* event );
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void setUnSelected();
    virtual void setSelected();

private:
    void setParameters();

    Imaging::InputPortAbstractImage*		_inPort;
    vtkIntegration::m4dImageDataSource*		_imageData;
    vtkImageCast*				_iCast;
    vtkPiecewiseFunction*			_opacityTransferFunction;
    vtkColorTransferFunction*			_colorTransferFunction;
    vtkVolumeProperty*				_volumeProperty;
    vtkVolumeRayCastMapper*			_volumeMapper;
    vtkVolume*					_volume;
    vtkActor2D*					_actor2D;
    vtkPoints*					_points;
    vtkPolyData*				_pointsData;
    vtkPolyDataMapper2D*			_pointsDataMapper;
    vtkCellArray*				_cells;
    vtkRenderer*				_renImageData;
    AvailableSlots				_availableSlots;
};

} /* namespace Viewer */
} /* namespace M4D */

#endif // M4D_GUI_VTK_VIEWER_WIDGET_H

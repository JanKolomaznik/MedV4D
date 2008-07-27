#include "GUI/m4dGUIVtkViewerWidget.h"

#include <QtGui>

namespace M4D
{
namespace Viewer
{

m4dGUIVtkViewerWidget::m4dGUIVtkViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent )
    : QVTKWidget( parent )
{
    _index = index;
    setParameters();
    _inPort = new Imaging::InputPortAbstractImage();
    _inputPorts.AddPort( _inPort );
    setInputPort( conn );
}

m4dGUIVtkViewerWidget::m4dGUIVtkViewerWidget( unsigned index, QWidget *parent )
    : QVTKWidget( parent )
{
    _index = index;
    setParameters();
    _inPort = new Imaging::InputPortAbstractImage();
    _inputPorts.AddPort( _inPort );
    setInputPort( );
}

m4dGUIVtkViewerWidget::~m4dGUIVtkViewerWidget()
{
    _imageData->Delete();
    _iCast->Delete();
    _opacityTransferFunction->Delete();
    _colorTransferFunction->Delete();
    _volumeProperty->Delete();
    _volumeMapper->Delete();
    _volume->Delete();
    _renImageData->Delete();
    _actor2D->Delete();
    _points->Delete();
    _pointsData->Delete();
    _pointsDataMapper->Delete();
    _cells->Delete();
}

void
m4dGUIVtkViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
{
    conn->ConnectConsumer( *_inPort );
    _imageData->TemporarySetImageData( _inPort->GetAbstractImage() );
}

void
m4dGUIVtkViewerWidget::setInputPort()
{
    _inPort->UnPlug();
    _imageData->TemporaryUnsetImageData();
}

void
m4dGUIVtkViewerWidget::setUnSelected()
{
    if ( _selected )
    {
        _renImageData->RemoveViewProp( _actor2D );
    }
    _selected = false;
}

void
m4dGUIVtkViewerWidget::setSelected()
{
    if ( !_selected )
    {
        _renImageData->AddViewProp( _actor2D );
    }
    _selected = true;
    emit signalSetSelected( _index, false );
}

void
m4dGUIVtkViewerWidget::resizeEvent( QResizeEvent* event )
{
  QVTKWidget::resizeEvent( event );
  _points->SetNumberOfPoints( 5 );
  _points->SetPoint(0,           3,            3, 0 );
  _points->SetPoint(1,           3, height() - 4, 0 );
  _points->SetPoint(2, width() - 4, height() - 4, 0 );
  _points->SetPoint(3, width() - 4,            3, 0 );
  _points->SetPoint(4,           3,            3, 0 );
  _cells->Reset();
  _cells->InsertNextCell(5);
  _cells->InsertCellPoint(0);
  _cells->InsertCellPoint(1);
  _cells->InsertCellPoint(2);
  _cells->InsertCellPoint(3);
  _cells->InsertCellPoint(4);
}

void
m4dGUIVtkViewerWidget::setParameters()
{
  
  _selected = false;

  _imageData = vtkIntegration::m4dImageDataSource::New();

  _iCast = vtkImageCast::New(); 
  _iCast->SetOutputScalarTypeToUnsignedShort();
  _iCast->SetInputConnection( _imageData->GetOutputPort() );

  _opacityTransferFunction = vtkPiecewiseFunction::New();
  _opacityTransferFunction->AddPoint( 0,   0.0 ); 	
  _opacityTransferFunction->AddPoint( 169, 0.0 );
  _opacityTransferFunction->AddPoint( 170, 0.2 );	
  _opacityTransferFunction->AddPoint( 400, 0.2 );
  _opacityTransferFunction->AddPoint( 401, 0.0 );

  _colorTransferFunction = vtkColorTransferFunction::New();
  _colorTransferFunction->AddRGBPoint(    0.0, 0.0, 0.0, 0.0 ); 
  _colorTransferFunction->AddRGBPoint(  170.0, 1.0, 0.0, 0.0 );
  _colorTransferFunction->AddRGBPoint(  400.0, 0.8, 0.8, 0.8 );
  _colorTransferFunction->AddRGBPoint( 2000.0, 1.0, 1.0, 1.0 );
  	
  _volumeProperty = vtkVolumeProperty::New();
  _volumeProperty->SetColor( _colorTransferFunction );
  _volumeProperty->SetScalarOpacity( _opacityTransferFunction );
  //volumeProperty->ShadeOn(); 
  _volumeProperty->SetInterpolationTypeToLinear();

  _volumeMapper = vtkVolumeRayCastMapper::New();
  _volumeMapper->SetVolumeRayCastFunction( vtkVolumeRayCastCompositeFunction::New());
  _volumeMapper->SetInputConnection( _iCast->GetOutputPort() );

  _volume = vtkVolume::New();
  _volume->SetMapper( _volumeMapper ); 
  _volume->SetProperty( _volumeProperty );

  _actor2D = vtkActor2D::New();
  _points = vtkPoints::New();
  _pointsData = vtkPolyData::New();
  _pointsDataMapper = vtkPolyDataMapper2D::New();
  _actor2D->GetProperty()->SetColor( 0., 1., 0. );
  _pointsDataMapper->SetInput(_pointsData);
  _pointsData->SetPoints(_points);
  _actor2D->SetMapper(_pointsDataMapper);
  _cells = vtkCellArray::New();
  _pointsData->SetLines(_cells);

  _renImageData = vtkRenderer::New(); 
  _renImageData->AddViewProp( _volume );

  vtkRenderWindow *rWin;
  rWin = GetRenderWindow();

  rWin->AddRenderer( _renImageData );

  vtkRenderWindowInteractor *iren;
  iren = GetInteractor();
  iren->SetRenderWindow( rWin );
}

m4dGUIVtkViewerWidget::AvailableSlots
m4dGUIVtkViewerWidget::getAvailableSlots()
{
    return _availableSlots;
}

QWidget*
m4dGUIVtkViewerWidget::operator()()
{
    return (QVTKWidget*)this;
}

void
m4dGUIVtkViewerWidget::ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction )
{
    switch( msg->msgID )
    {
	case Imaging::PMI_FILTER_UPDATED:
	//updateGL();
	break;

	default:
	break;
    }
}

void
m4dGUIVtkViewerWidget::slotSetButtonHandler( ButtonHandler hnd, MouseButton btn ) {}

void
m4dGUIVtkViewerWidget::slotSetSelected( bool selected )
{
    if ( selected ) setSelected();
    else setUnSelected();
}

void
m4dGUIVtkViewerWidget::slotSetSliceNum( size_t num ) {}

void
m4dGUIVtkViewerWidget::slotSetOneSliceMode() {}

void
m4dGUIVtkViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow ) {}

void
m4dGUIVtkViewerWidget::slotToggleFlipVertical() {}

void
m4dGUIVtkViewerWidget::slotToggleFlipHorizontal() {}

void
m4dGUIVtkViewerWidget::slotAddLeftSideData( std::string type, std::string data ) {}

void
m4dGUIVtkViewerWidget::slotAddRightSideData( std::string type, std::string data ) {}

void
m4dGUIVtkViewerWidget::slotEraseLeftSideData( std::string type ) {}

void
m4dGUIVtkViewerWidget::slotEraseRightSideData( std::string type ) {}

void
m4dGUIVtkViewerWidget::slotClearLeftSideData() {}

void
m4dGUIVtkViewerWidget::slotClearRightSideData() {}

void
m4dGUIVtkViewerWidget::slotTogglePrintData() {}

void
m4dGUIVtkViewerWidget::slotZoom( int amount ) {}

void
m4dGUIVtkViewerWidget::slotMove( int amountH, int amountV ) {}

void
m4dGUIVtkViewerWidget::slotAdjustContrastBrightness( int amountB, int amountC ) {}

void
m4dGUIVtkViewerWidget::slotNewPoint( int x, int y, int z ) {}

void
m4dGUIVtkViewerWidget::slotNewShape( int x, int y, int z ) {}

void
m4dGUIVtkViewerWidget::slotDeletePoint() {}

void
m4dGUIVtkViewerWidget::slotDeleteShape() {}

void
m4dGUIVtkViewerWidget::slotDeleteAll() {}

void
m4dGUIVtkViewerWidget::slotRotateAxisX( int x ) {}

void
m4dGUIVtkViewerWidget::slotRotateAxisY( int y ) {}

void
m4dGUIVtkViewerWidget::slotRotateAxisZ( int z ) {}

} /* namespace Viewer */
} /* namespace M4D */

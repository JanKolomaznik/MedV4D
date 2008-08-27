/**
 *  @ingroup gui
 *  @file m4dGUIVtkViewerWidget.cpp
 *  @brief ...
 */
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
    _actor2DSelected->Delete();
    _pointsSelected->Delete();
    _pointsDataSelected->Delete();
    _pointsDataMapperSelected->Delete();
    _cellsSelected->Delete();
    _actor2DPlugged->Delete();
    _pointsPlugged->Delete();
    _pointsDataPlugged->Delete();
    _pointsDataMapperPlugged->Delete();
    _cellsPlugged->Delete();
}

void
m4dGUIVtkViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
{
    if ( !conn )
    {
        setInputPort();
	return;
    }
    _renImageData->RemoveViewProp( _volume );
    GetRenderWindow()->RemoveRenderer( _renImageData );
    _renImageData->Delete();
    _renImageData = vtkRenderer::New();
    conn->ConnectConsumer( *_inPort );
    try
    {
        if ( _inPort->TryLockDataset() )
	{
            try
	    {
                if ( _inPort->GetAbstractImage().GetDimension() == 3 ) 
                    _imageData->TemporarySetImageData( _inPort->GetAbstractImage() );
	        _inPort->ReleaseDatasetLock();
	    } catch (...) {}
	}
    } catch (...) {}
    _renImageData->AddViewProp( _volume );
    GetRenderWindow()->AddRenderer( _renImageData );
    if ( _selected ) _renImageData->AddViewProp( _actor2DSelected );
    _renImageData->AddViewProp( _actor2DPlugged );
    GetRenderWindow()->Render();
    _plugged = true;

}

void
m4dGUIVtkViewerWidget::setInputPort()
{
    if ( _inPort->IsPlugged() )
    {
        _renImageData->RemoveViewProp( _actor2DPlugged );
    }
    _inPort->UnPlug();
    _imageData->TemporaryUnsetImageData();
    GetRenderWindow()->Render();
    _plugged = false;
}

void
m4dGUIVtkViewerWidget::setUnSelected()
{
    if ( _selected )
    {
        _renImageData->RemoveViewProp( _actor2DSelected );
	GetRenderWindow()->Render();
    }
    _selected = false;
}

void
m4dGUIVtkViewerWidget::setSelected()
{
    if ( !_selected )
    {
        _renImageData->AddViewProp( _actor2DSelected );
	GetRenderWindow()->Render();
    }
    _selected = true;
    emit signalSetSelected( _index, false );
}

void
m4dGUIVtkViewerWidget::setBorderPoints( vtkPoints* points, vtkCellArray *cells, unsigned pos )
{
  points->SetNumberOfPoints( 5 );
  points->SetPoint(0,               pos,                pos, 0 );
  points->SetPoint(1,               pos, height() - 1 - pos, 0 );
  points->SetPoint(2, width() - 1 - pos, height() - 1 - pos, 0 );
  points->SetPoint(3, width() - 1 - pos,                pos, 0 );
  points->SetPoint(4,               pos,                pos, 0 );
  cells->Reset();
  cells->InsertNextCell(5);
  cells->InsertCellPoint(0);
  cells->InsertCellPoint(1);
  cells->InsertCellPoint(2);
  cells->InsertCellPoint(3);
  cells->InsertCellPoint(4);
}

void
m4dGUIVtkViewerWidget::resizeEvent( QResizeEvent* event )
{
  QVTKWidget::resizeEvent( event );
  setBorderPoints( _pointsSelected, _cellsSelected, 1 );
  setBorderPoints(  _pointsPlugged,  _cellsPlugged, 2 );
  GetRenderWindow()->Render();
}

void
m4dGUIVtkViewerWidget::mousePressEvent(QMouseEvent *event)
{
    if ( !_selected ) setSelected();
    else QVTKWidget::mousePressEvent( event );
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

  _actor2DSelected = vtkActor2D::New();
  _pointsSelected = vtkPoints::New();
  _pointsDataSelected = vtkPolyData::New();
  _pointsDataMapperSelected = vtkPolyDataMapper2D::New();
  _actor2DSelected->GetProperty()->SetColor( 0., 1., 0. );
  _pointsDataMapperSelected->SetInput(_pointsDataSelected);
  _pointsDataSelected->SetPoints(_pointsSelected);
  _actor2DSelected->SetMapper(_pointsDataMapperSelected);
  _cellsSelected = vtkCellArray::New();
  _pointsDataSelected->SetLines(_cellsSelected);
  
  _actor2DPlugged = vtkActor2D::New();
  _pointsPlugged = vtkPoints::New();
  _pointsDataPlugged = vtkPolyData::New();
  _pointsDataMapperPlugged = vtkPolyDataMapper2D::New();
  _actor2DPlugged->GetProperty()->SetColor( 0., 0., 1. );
  _pointsDataMapperPlugged->SetInput(_pointsDataPlugged);
  _pointsDataPlugged->SetPoints(_pointsPlugged);
  _actor2DPlugged->SetMapper(_pointsDataMapperPlugged);
  _cellsPlugged = vtkCellArray::New();
  _pointsDataPlugged->SetLines(_cellsPlugged);

  _renImageData = vtkRenderer::New(); 

  _renImageData->AddViewProp( _volume );
  GetRenderWindow()->AddRenderer( _renImageData );
  
  vtkRenderWindow *rWin;
  rWin = GetRenderWindow();

  vtkRenderWindowInteractor *iren;
  iren = GetInteractor();
  iren->SetRenderWindow( rWin );

  _availableSlots.clear();
  _availableSlots.push_back( SETSELECTED );
  _availableSlots.push_back( ZOOM );
  _availableSlots.push_back( ROTATEAXISX );
  _availableSlots.push_back( ROTATEAXISY );
  _availableSlots.push_back( ROTATEAXISZ );
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
    emit signalMessageHandler( msg->msgID );
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
m4dGUIVtkViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn ) {}

void
m4dGUIVtkViewerWidget::slotToggleFlipVertical() {}

void
m4dGUIVtkViewerWidget::slotToggleFlipHorizontal() {}

void
m4dGUIVtkViewerWidget::slotAddLeftSideData( std::string data ) {}

void
m4dGUIVtkViewerWidget::slotAddRightSideData( std::string data ) {}

void
m4dGUIVtkViewerWidget::slotClearLeftSideData() {}

void
m4dGUIVtkViewerWidget::slotClearRightSideData() {}

void
m4dGUIVtkViewerWidget::slotTogglePrintData() {}

void
m4dGUIVtkViewerWidget::slotTogglePrintShapeData() {}

void
m4dGUIVtkViewerWidget::slotZoom( int amount )
{
    _renImageData->GetActiveCamera()->Zoom( ((double)amount)/10. );
}

void
m4dGUIVtkViewerWidget::slotMove( int amountH, int amountV ) {}

void
m4dGUIVtkViewerWidget::slotAdjustContrastBrightness( int amountB, int amountC ) {}

void
m4dGUIVtkViewerWidget::slotNewPoint( double x, double y, double z ) {}

void
m4dGUIVtkViewerWidget::slotNewShape( double x, double y, double z ) {}

void
m4dGUIVtkViewerWidget::slotDeletePoint() {}

void
m4dGUIVtkViewerWidget::slotDeleteShape() {}

void
m4dGUIVtkViewerWidget::slotDeleteAll() {}

void
m4dGUIVtkViewerWidget::slotRotateAxisX( double x )
{
    _renImageData->GetActiveCamera()->Elevation( x );
}

void
m4dGUIVtkViewerWidget::slotRotateAxisY( double y )
{
    _renImageData->GetActiveCamera()->Azimuth( y );
}

void
m4dGUIVtkViewerWidget::slotRotateAxisZ( double z )
{
    _renImageData->GetActiveCamera()->Roll( z );
}

void
m4dGUIVtkViewerWidget::slotToggleSliceOrientation()
{
}

void
m4dGUIVtkViewerWidget::slotColorPicker( double x, double y, double z )
{
}

void
m4dGUIVtkViewerWidget::slotMessageHandler( Imaging::PipelineMsgID msgID )
{
    switch( msgID )
    {
	case Imaging::PMI_FILTER_UPDATED:
        case Imaging::PMI_DATASET_PUT:
	case Imaging::PMI_PORT_PLUGGED:
        _renImageData->RemoveViewProp( _volume );
        GetRenderWindow()->RemoveRenderer( _renImageData );
        _renImageData->Delete();
        _renImageData = vtkRenderer::New();
        try
        {
	    if ( _inPort->TryLockDataset() )
	    {
                try
	        {
	            if ( _inPort->GetAbstractImage().GetDimension() == 3 ) 
                        _imageData->TemporarySetImageData( _inPort->GetAbstractImage() );
	        } catch (...) {}
	        _inPort->ReleaseDatasetLock();
	    }
        } catch (...) {}
	_renImageData->AddViewProp( _volume );
        GetRenderWindow()->AddRenderer( _renImageData );
        if ( _selected ) _renImageData->AddViewProp( _actor2DSelected );
        _renImageData->AddViewProp( _actor2DPlugged );
	_plugged = true;
	break;

	default:
	break;
    }
    GetRenderWindow()->Render();
}

} /* namespace Viewer */
} /* namespace M4D */

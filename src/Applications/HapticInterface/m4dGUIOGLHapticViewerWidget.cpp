/**
*  @ingroup gui
*  @file m4dGUIOpenGLViewerWidget.cpp
*  @brief some brief
*/

#include "m4dGUIOGLHapticViewerWidget.h"
#include <QtGui>

namespace M4D
{
	namespace Viewer
	{
		m4dGUIOGLHapticViewerWidget::tissue::tissue(vtkAlgorithmOutput* data, double val, double r, double g, double b, double opacity)
		{
			iso = vtkMarchingCubes::New();
			iso->SetInputConnection(data);
			//iso->GenerateValues(1, val, val + 400.0);
			iso->SetValue(0, val);

			isoNormals = vtkPolyDataNormals::New();
			isoNormals->SetInputConnection(iso->GetOutputPort());
			isoNormals->SetFeatureAngle(179.0);

			isoMapper = vtkPolyDataMapper::New();
			isoMapper->SetInput(isoNormals->GetOutput());
			isoMapper->ScalarVisibilityOff();

			isoActor = vtkActor::New();
			isoActor->SetMapper(isoMapper);
			isoActor->GetProperty()->SetColor(r, g, b);
			isoActor->GetProperty()->SetOpacity(opacity);

		}
		vtkActor* m4dGUIOGLHapticViewerWidget::tissue::GetActor()
		{
			return isoActor;
		}
		void m4dGUIOGLHapticViewerWidget::tissue::deleteInnerItems()
		{
			iso->Delete();
			isoNormals->Delete();
			isoMapper->Delete();
			isoActor->Delete();
		}
		m4dGUIOGLHapticViewerWidget::m4dGUIOGLHapticViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent )
			: QVTKWidget( parent )
		{
			_index = index;
			setParameters();
			_inPort = new Imaging::InputPortTyped< Imaging::AImage >();
			_inputPorts.AppendPort( _inPort );
			setInputPort( conn );
		}

		m4dGUIOGLHapticViewerWidget::m4dGUIOGLHapticViewerWidget( unsigned index, QWidget *parent )
			: QVTKWidget( parent )
		{
			_index = index;
			setParameters();
			_inPort = new Imaging::InputPortTyped< Imaging::AImage >();
			_inputPorts.AppendPort( _inPort );
			setInputPort( );
		}

		m4dGUIOGLHapticViewerWidget::~m4dGUIOGLHapticViewerWidget()
		{
			_imageData->Delete();
			_iCast->Delete();
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
			std::vector< tissue >::iterator it;
			for (it = tissues.begin(); it != tissues.end(); it++)
			{
				(*it).deleteInnerItems();
			}
		}

		void m4dGUIOGLHapticViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
		{
			if ( !conn )
			{
				setInputPort();
				return;
			}
			std::vector< tissue >::iterator it;
			for (it = tissues.begin(); it != tissues.end(); it++)
			{
				_renImageData->RemoveActor((*it).GetActor());
			}
			_renImageData->RemoveActor(cursorActor);
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
						if ( _inPort->GetDatasetTyped().GetDimension() == 3 ) 
							_imageData->TemporarySetImageData( _inPort->GetDatasetTyped() );
						_inPort->ReleaseDatasetLock();
					} catch (...) {}
				}
			} catch (...) {}
			for (it = tissues.begin(); it != tissues.end(); it++)
			{
				_renImageData->AddActor((*it).GetActor());
			}
			_renImageData->AddActor(cursorActor);
			GetRenderWindow()->AddRenderer( _renImageData );
			//if ( _selected ) _renImageData->AddViewProp( _actor2DSelected );
			//_renImageData->AddViewProp( _actor2DPlugged );
			GetRenderWindow()->Render();
			_plugged = true;

		}

		void m4dGUIOGLHapticViewerWidget::setInputPort()
		{
			if ( _inPort->IsPlugged() )
			{
				//_renImageData->RemoveViewProp( _actor2DPlugged );
			}
			_inPort->UnPlug();
			_imageData->TemporaryUnsetImageData();
			GetRenderWindow()->Render();
			_plugged = false;
		}

		void m4dGUIOGLHapticViewerWidget::setUnSelected()
		{
			if ( _selected )
			{
				//_renImageData->RemoveViewProp( _actor2DSelected );
				GetRenderWindow()->Render();
			}
			_selected = false;
		}

		void m4dGUIOGLHapticViewerWidget::setSelected()
		{
			if ( !_selected )
			{
				//_renImageData->AddViewProp( _actor2DSelected );
				GetRenderWindow()->Render();
			}
			_selected = true;
			emit signalSetSelected( _index, false );
		}

		void m4dGUIOGLHapticViewerWidget::setBorderPoints( vtkPoints* points, vtkCellArray *cells, unsigned pos )
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

		void m4dGUIOGLHapticViewerWidget::resizeEvent( QResizeEvent* event )
		{
			QVTKWidget::resizeEvent( event );
			setBorderPoints( _pointsSelected, _cellsSelected, 1 );
			setBorderPoints(  _pointsPlugged,  _cellsPlugged, 2 );
			GetRenderWindow()->Render();
		}

		void m4dGUIOGLHapticViewerWidget::mousePressEvent(QMouseEvent *event)
		{
			if ( !_selected ) setSelected();
			else if ( _eventHandler ) _eventHandler->mouseDoubleClickEvent(event);
			else QVTKWidget::mousePressEvent( event );
		}

		void m4dGUIOGLHapticViewerWidget::mouseDoubleClickEvent ( QMouseEvent * event )
		{
			if ( _eventHandler ) _eventHandler->mouseDoubleClickEvent(event);
			else QVTKWidget::mouseDoubleClickEvent( event );
		}

		void m4dGUIOGLHapticViewerWidget::mouseMoveEvent ( QMouseEvent * event )
		{
			if ( _eventHandler ) _eventHandler->mouseMoveEvent(event);
			else QVTKWidget::mouseMoveEvent( event );
		}

		void m4dGUIOGLHapticViewerWidget::mouseReleaseEvent ( QMouseEvent * event )
		{
			if ( _eventHandler ) _eventHandler->mouseReleaseEvent(event);
			else QVTKWidget::mouseReleaseEvent( event );
		}

		void m4dGUIOGLHapticViewerWidget::wheelEvent ( QWheelEvent * event )
		{
			if ( _eventHandler ) _eventHandler->wheelEvent(event);
			else QVTKWidget::wheelEvent( event );
		}

		void m4dGUIOGLHapticViewerWidget::keyPressEvent ( QKeyEvent * event )
		{
			if ( _eventHandler ) _eventHandler->keyPressEvent(event);
			else QVTKWidget::keyPressEvent( event );
		}

		void m4dGUIOGLHapticViewerWidget::keyReleaseEvent ( QKeyEvent * event )
		{
			if ( _eventHandler ) _eventHandler->keyReleaseEvent(event);
			else QVTKWidget::keyReleaseEvent( event );
		}
			
		void m4dGUIOGLHapticViewerWidget::setParameters()
		{

			_selected = false;
			_imageData = vtkIntegration::m4dImageDataSource::New();

			_iCast = vtkImageCast::New(); 
			_iCast->SetOutputScalarTypeToUnsignedShort();
			_iCast->SetInputConnection( _imageData->GetOutputPort() );

			std::cout << "Set marching cubes..." << std::endl; // DEBUG

			tissues.push_back(tissue(_iCast->GetOutputPort(), 600, 1.0, 1.0, 0.4, 0.35)); // Lungs and skin
			//tissues.push_back(tissue(_iCast->GetOutputPort(), 900, 0.5, 0.5, 1.0, 0.45)); // soft tissue
			//tissues.push_back(tissue(_iCast->GetOutputPort(), 1080, 1.0, 0.5, 0.5, 0.5)); // Muscles
			//tissues.push_back(tissue(_iCast->GetOutputPort(), 1300, 0.8, 0.8, 0.8, 1.0)); // Bones

			std::cout << "Old..." << std::endl; // DEBUG

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

			cursorSphere = vtkSphereSource::New();
			cursorSphere->SetCenter(0.0, 0.0, 0.0);
			cursorSphere->SetRadius(2.0);

			cursorMapper = vtkPolyDataMapper::New();
			cursorMapper->SetInput(cursorSphere->GetOutput());

			cursorActor = vtkActor::New();
			cursorActor->SetMapper(cursorMapper);
			cursorActor->GetProperty()->SetColor(0.0, 1.0, 0.0);
			cursorActor->GetProperty()->SetOpacity(1.0);
			cursorActor->SetPosition(380.0, 100.0, 200.0);

			std::cout << "Create renderer..." << std::endl; // DEBUG

			_renImageData = vtkRenderer::New(); 

			GetRenderWindow()->AddRenderer( _renImageData );

			std::vector< tissue >::iterator it;
			for (it = tissues.begin(); it != tissues.end(); it++)
			{
				_renImageData->AddActor((*it).GetActor());
			}
			_renImageData->AddActor(cursorActor);

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

			

			std::cout << "End set parameters..." << std::endl; // DEBUG
		}

		m4dGUIOGLHapticViewerWidget::AvailableSlots
			m4dGUIOGLHapticViewerWidget::getAvailableSlots()
		{
			return _availableSlots;
		}

		QWidget*
			m4dGUIOGLHapticViewerWidget::operator()()
		{
			return (QVTKWidget*)this;
		}

		void m4dGUIOGLHapticViewerWidget::ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction )
		{
			emit signalMessageHandler( msg->msgID );
		}

		void
			m4dGUIOGLHapticViewerWidget::slotSetButtonHandler( ButtonHandler hnd, MouseButton btn ) {}

		void m4dGUIOGLHapticViewerWidget::slotSetSelected( bool selected )
		{
			if ( selected ) setSelected();
			else setUnSelected();
		}

		void m4dGUIOGLHapticViewerWidget::slotSetSliceNum( size_t num ) {}

		void m4dGUIOGLHapticViewerWidget::slotSetOneSliceMode() {}

		void m4dGUIOGLHapticViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn ) {}

		void m4dGUIOGLHapticViewerWidget::slotToggleFlipVertical() {}

		void m4dGUIOGLHapticViewerWidget::slotToggleFlipHorizontal() {}

		void m4dGUIOGLHapticViewerWidget::slotAddLeftSideData( std::string data ) {}

		void m4dGUIOGLHapticViewerWidget::slotAddRightSideData( std::string data ) {}

		void m4dGUIOGLHapticViewerWidget::slotClearLeftSideData() {}

		void m4dGUIOGLHapticViewerWidget::slotClearRightSideData() {}

		void m4dGUIOGLHapticViewerWidget::slotTogglePrintData() {}

		void m4dGUIOGLHapticViewerWidget::slotTogglePrintShapeData() {}

		void m4dGUIOGLHapticViewerWidget::slotZoom( int amount )
		{
			_renImageData->GetActiveCamera()->Zoom( ((double)amount)/10. );
		}

		void m4dGUIOGLHapticViewerWidget::slotMove( int amountH, int amountV ) {}

		void m4dGUIOGLHapticViewerWidget::slotAdjustContrastBrightness( int amountB, int amountC ) {}

		void m4dGUIOGLHapticViewerWidget::slotNewPoint( double x, double y, double z ) {}

		void m4dGUIOGLHapticViewerWidget::slotNewShape( double x, double y, double z ) {}

		void m4dGUIOGLHapticViewerWidget::slotDeletePoint() {}

		void m4dGUIOGLHapticViewerWidget::slotDeleteShape() {}

		void m4dGUIOGLHapticViewerWidget::slotDeleteAll() {}

		void m4dGUIOGLHapticViewerWidget::slotRotateAxisX( double x )
		{
			_renImageData->GetActiveCamera()->Elevation( x );
		}

		void m4dGUIOGLHapticViewerWidget::slotRotateAxisY( double y )
		{
			_renImageData->GetActiveCamera()->Azimuth( y );
		}

		void m4dGUIOGLHapticViewerWidget::slotRotateAxisZ( double z )
		{
			_renImageData->GetActiveCamera()->Roll( z );
		}

		void m4dGUIOGLHapticViewerWidget::slotToggleSliceOrientation()
		{
		}

		void m4dGUIOGLHapticViewerWidget::slotColorPicker( double x, double y, double z )
		{
		}

		void m4dGUIOGLHapticViewerWidget::slotMessageHandler( Imaging::PipelineMsgID msgID )
		{
			std::vector< tissue >::iterator it;
			switch( msgID )
			{
			case Imaging::PMI_FILTER_UPDATED:
			case Imaging::PMI_DATASET_PUT:
			case Imaging::PMI_PORT_PLUGGED:
				for (it = tissues.begin(); it != tissues.end(); it++)
				{
					_renImageData->RemoveActor((*it).GetActor());
				}
				_renImageData->RemoveActor(cursorActor);
				GetRenderWindow()->RemoveRenderer( _renImageData );
				_renImageData->Delete();
				_renImageData = vtkRenderer::New();
				try
				{
					if ( _inPort->TryLockDataset() )
					{
						try
						{
							if ( _inPort->GetDatasetTyped().GetDimension() == 3 ) 
								_imageData->TemporarySetImageData( _inPort->GetDatasetTyped() );
						} catch (...) {}
						_inPort->ReleaseDatasetLock();
					}
				} catch (...) {}
				for (it = tissues.begin(); it != tissues.end(); it++)
				{
					_renImageData->AddActor((*it).GetActor());
				}
				_renImageData->AddActor(cursorActor);
				GetRenderWindow()->AddRenderer( _renImageData );
				//if ( _selected ) _renImageData->AddViewProp( _actor2DSelected );
				//_renImageData->AddViewProp( _actor2DPlugged );
				_plugged = true;
				break;

			default:
				break;
			}
			GetRenderWindow()->Render();
		}

	} /* namespace Viewer */
} /* namespace M4D */
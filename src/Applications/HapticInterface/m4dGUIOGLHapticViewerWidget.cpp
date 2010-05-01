/**
*  @ingroup gui
*  @file m4dGUIOpenGLViewerWidget.cpp
*  @brief some brief
*/

#include "m4dGUIOGLHapticViewerWidget.h"
#include <QtGui>
#include <QWidget>

namespace M4D
{
	namespace Viewer
	{
		
		m4dGUIOGLHapticViewerWidget::tissue::tissue(vtkAlgorithmOutput* data, double val, double r, double g, double b, double opacity)
		{
			iso = vtkMarchingCubes::New();
			iso->SetInputConnection(data);
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
		m4dGUIOGLHapticViewerWidget::line::line()
		{
			m_source = vtkLineSource::New();
			double p[3] = {0.0, 0.0, 0.0};
			m_source->SetPoint1(p);
			m_source->SetPoint2(p);

			m_mapper = vtkPolyDataMapper2D::New();
			m_mapper->SetInput(m_source->GetOutput());

			m_actor = vtkActor2D::New();
			m_actor->SetMapper(m_mapper);
			m_actor->SetLayerNumber(1);
		}
		m4dGUIOGLHapticViewerWidget::line::~line()
		{
			m_source->Delete();
			m_mapper->Delete();
			m_actor->Delete();
		}

		void m4dGUIOGLHapticViewerWidget::line::SetPoints( double a_point0[3], double a_point1[3] )
		{
			m_source->SetPoint1(a_point0);
			m_source->SetPoint2(a_point1);
		}

		void m4dGUIOGLHapticViewerWidget::line::SetColor( double a_red, double a_green, double a_blue )
		{
			m_actor->GetProperty()->SetColor(a_red, a_green, a_blue);
		}

		vtkActor2D* m4dGUIOGLHapticViewerWidget::line::GetActor()
		{
			return m_actor;
		}

		m4dGUIOGLHapticViewerWidget::m4dGUIOGLHapticViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent )
			: QVTKWidget( parent )
		{
			_index = index;
			hapticForceTransitionFunction = new transitionFunction();
			setParameters();
			_inPort = new Imaging::InputPortTyped< Imaging::AImage >();
			_inputPorts.AppendPort( _inPort );
			setInputPort( conn );
			aggregationFilter->Update();
			cursor = new hapticCursor(aggregationFilter->GetOutput(), GetRenderWindow(), hapticForceTransitionFunction);
			reloadCursorParameters();
			cursor->startHaptics();
			QTimer *timer = new QTimer((QVTKWidget*)this);
			QObject::connect(timer, SIGNAL(timeout()), (QVTKWidget*)this, SLOT(update()));
			timer->start(18);
			settings = new SettingsBoxWidget(hapticForceTransitionFunction);
			QObject::connect((QVTKWidget*)this, SIGNAL(hapticForceTransitionFunctionChanged()), settings, SLOT(functionChangedSlot()));
			QObject::connect(settings, SIGNAL(resetFunction()), (QVTKWidget*)this, SLOT(slotTransitionFunctionResetDemanded()));
			QObject::connect(settings, SIGNAL(zoomInHaptic()), (QVTKWidget*)this, SLOT(slotZoomInHaptic()));
			QObject::connect(settings, SIGNAL(zoomOutHaptic()), (QVTKWidget*)this, SLOT(slotZoomOutHaptic()));
			settings->show();
			resetTransitionFunction();
			resetSliceViewPosition();
		}

		m4dGUIOGLHapticViewerWidget::m4dGUIOGLHapticViewerWidget( unsigned index, QWidget *parent )
			: QVTKWidget( parent )
		{
			_index = index;
			hapticForceTransitionFunction = new transitionFunction();
			setParameters();
			_inPort = new Imaging::InputPortTyped< Imaging::AImage >();
			_inputPorts.AppendPort( _inPort );
			setInputPort();
			cursor = new hapticCursor(aggregationFilter->GetOutput(), GetRenderWindow(), hapticForceTransitionFunction);
			reloadCursorParameters();
			cursor->startHaptics();
			QTimer *timer = new QTimer((QVTKWidget*)this);
			QObject::connect(timer, SIGNAL(timeout()), (QVTKWidget*)this, SLOT(update()));
			timer->start(18);
			settings = new SettingsBoxWidget(hapticForceTransitionFunction);
			QObject::connect((QVTKWidget*)this, SIGNAL(hapticForceTransitionFunctionChanged()), settings, SLOT(functionChangedSlot()));
			QObject::connect(settings, SIGNAL(resetFunction()), (QVTKWidget*)this, SLOT(slotTransitionFunctionResetDemanded()));
			settings->show();
			resetTransitionFunction();
			resetSliceViewPosition();
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
			cursorCubeExtractEdges->Delete();
			cursorCubeMapper->Delete();
			cursorCubeActor->Delete();
			sliceMapper->Delete();
			sliceActor->Delete();
			sliceRenderer->Delete();
			std::vector< tissue >::iterator it;
			for (it = tissues.begin(); it != tissues.end(); it++)
			{
				(*it).deleteInnerItems();
			}
			delete(cursor);
			delete(hapticForceTransitionFunction);
			delete(settings);
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
			_renImageData->RemoveActor(cursorCubeActor);
			sliceRenderer->RemoveActor(sliceActor);
			sliceRenderer->RemoveActor(rectangleLine1.GetActor());
			sliceRenderer->RemoveActor(rectangleLine2.GetActor());
			sliceRenderer->RemoveActor(rectangleLine3.GetActor());
			sliceRenderer->RemoveActor(rectangleLine4.GetActor());
			sliceRenderer->RemoveActor(cursorLine1.GetActor());
			sliceRenderer->RemoveActor(cursorLine2.GetActor());
			GetRenderWindow()->RemoveRenderer( _renImageData );
			GetRenderWindow()->RemoveRenderer(sliceRenderer);
			sliceRenderer->Delete();
			sliceRenderer = vtkRenderer::New();
			_renImageData->Delete();
			_renImageData = vtkOpenGLRenderer::New();
			_renImageData->SetUseDepthPeeling(1);  //This and next 2 lines enables depth peeling
			_renImageData->SetMaximumNumberOfPeels(50);
			_renImageData->SetOcclusionRatio(0.2);
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
			_renImageData->AddActor(cursorCubeActor);
			sliceRenderer->AddActor(sliceActor);
			sliceRenderer->AddActor(rectangleLine1.GetActor());
			sliceRenderer->AddActor(rectangleLine2.GetActor());
			sliceRenderer->AddActor(rectangleLine3.GetActor());
			sliceRenderer->AddActor(rectangleLine4.GetActor());
			sliceRenderer->AddActor(cursorLine1.GetActor());
			sliceRenderer->AddActor(cursorLine2.GetActor());
			GetRenderWindow()->AddRenderer( _renImageData );
			GetRenderWindow()->AddRenderer(sliceRenderer);
			//if ( _selected ) _renImageData->AddViewProp( _actor2DSelected );
			//_renImageData->AddViewProp( _actor2DPlugged );
			_plugged = true;
			double leftViewport[4] = {0.0, 0.0, 0.5, 1.0};
			double rightViewport[4] = {0.5, 0.0, 1.0, 1.0};

			_renImageData->SetViewport(leftViewport);
			sliceRenderer->SetViewport(rightViewport);
			GetRenderWindow()->Render();
			resetTransitionFunction();
		}

		void m4dGUIOGLHapticViewerWidget::setInputPort()
		{
			if ( _inPort->IsPlugged() )
			{
				//_renImageData->RemoveViewProp( _actor2DPlugged );
			}
			_inPort->UnPlug();
			_imageData->TemporaryUnsetImageData();
			_plugged = false;
		}

		void m4dGUIOGLHapticViewerWidget::setUnSelected()
		{
			if ( _selected )
			{
				//_renImageData->RemoveViewProp( _actor2DSelected );
			}
			_selected = false;
		}

		void m4dGUIOGLHapticViewerWidget::setSelected()
		{
			if ( !_selected )
			{
				//_renImageData->AddViewProp( _actor2DSelected );
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

		void m4dGUIOGLHapticViewerWidget::reloadCursorParameters()
		{
			double cursorCenter[3];
			cursor->GetCursorCenter(cursorCenter);
			cursorSource->SetCenter(cursorCenter);

			double cubeCenter[3];
			double scale = cursor->GetScale();
			cursor->GetRadiusCubeCenter(cubeCenter);
			cursorRadiusCube->SetCenter(cubeCenter);
			cursorRadiusCube->SetXLength(scale);
			cursorRadiusCube->SetYLength(scale);
			cursorRadiusCube->SetZLength(scale);

			sliceMapper->SetZSlice(cursor->GetZSlice());

			double point0[3], point1[3];
			int cursorCenterInt[3];
			cursor->GetCursorCenterAsIndexes(cursorCenterInt);
			int scaleInt = cursor->GetScaleAsIndexDifference();
			int cubeCenterInt[3];
			cursor->GetRadiusCubeCenterAsIndexes(cubeCenterInt);

			point0[0] = cursorCenterInt[0] - 6;
			point0[1] = cursorCenterInt[1];
			point0[2] = 0.0;
			point1[0] = cursorCenterInt[0] + 5;
			point1[1] = cursorCenterInt[1];
			point1[2] = 0.0;
			cursorLine1.SetPoints(point0, point1);
			
			point0[1] = cursorCenterInt[1] - 6;
			point0[0] = cursorCenterInt[0];
			point1[1] = cursorCenterInt[1] + 5;
			point1[0] = cursorCenterInt[0];
			cursorLine2.SetPoints(point0, point1);

			point0[0] = cubeCenterInt[0] - (scaleInt / 2);
			point0[1] = cubeCenterInt[1] - (scaleInt / 2);
			point1[0] = cubeCenterInt[0] + (scaleInt / 2);
			point1[1] = cubeCenterInt[1] - (scaleInt / 2);
			rectangleLine1.SetPoints(point0, point1);

			point0[0] = cubeCenterInt[0] + (scaleInt / 2);
			point0[1] = cubeCenterInt[1] - (scaleInt / 2);
			point1[0] = cubeCenterInt[0] + (scaleInt / 2);
			point1[1] = cubeCenterInt[1] + (scaleInt / 2);
			rectangleLine2.SetPoints(point0, point1);

			point0[0] = cubeCenterInt[0] - (scaleInt / 2);
			point0[1] = cubeCenterInt[1] + (scaleInt / 2);
			point1[0] = cubeCenterInt[0] + (scaleInt / 2);
			point1[1] = cubeCenterInt[1] + (scaleInt / 2);
			rectangleLine3.SetPoints(point0, point1);

			point0[0] = cubeCenterInt[0] - (scaleInt / 2);
			point0[1] = cubeCenterInt[1] + (scaleInt / 2);
			point1[0] = cubeCenterInt[0] - (scaleInt / 2);
			point1[1] = cubeCenterInt[1] - (scaleInt / 2);
			rectangleLine4.SetPoints(point0, point1);
		}
			
		void m4dGUIOGLHapticViewerWidget::setParameters()
		{

			_selected = false;
			_imageData = vtkIntegration::m4dImageDataSource::New();

			_iCast = vtkImageCast::New(); 
			_iCast->SetOutputScalarTypeToUnsignedShort();
			_iCast->SetInputConnection( _imageData->GetOutputPort() );

			std::cout << "Aggregation data filter setting..." << std::endl; // DEBUG

			aggregationFilter = aggregationFilterForVtk::New();
			aggregationFilter->SetAggregationPoint(550, 700, 600);
			//aggregationFilter->SetAggregationPoint(701, 1000, 900);
			//aggregationFilter->SetAggregationPoint(1001, 1200, 1080);
			//aggregationFilter->SetAggregationPoint(1201, 2000, 1500);
			aggregationFilter->SetInputConnection( _iCast->GetOutputPort());

			std::cout << "Set marching cubes..." << std::endl; // DEBUG

			tissues.push_back(tissue(aggregationFilter->GetOutputPort(), 600, 1.0, 1.0, 0.4, 0.35)); // Lungs and skin
			//tissues.push_back(tissue(aggregationFilter->GetOutputPort(), 900, 0.5, 0.5, 1.0, 0.4)); // soft tissue
			//tissues.push_back(tissue(aggregationFilter->GetOutputPort(), 1080, 1.0, 0.5, 0.5, 0.5)); // Muscles
			//tissues.push_back(tissue(aggregationFilter->GetOutputPort(), 1500, 0.8, 0.8, 0.8, 1.0)); // Bones

#pragma region pointStuffFromOriginalVtkViewer

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

#pragma endregion pointStuffFromOriginalVtkViewer

#pragma region cursorSourceVtkInicialization

			cursorSource = vtkSphereSource::New();
			cursorSource->SetCenter(0.0, 0.0, 0.0);
			cursorSource->SetRadius(1.0);

			cursorMapper = vtkPolyDataMapper::New();
			cursorMapper->SetInput(cursorSource->GetOutput());

			cursorActor = vtkActor::New();
			cursorActor->SetMapper(cursorMapper);
			cursorActor->GetProperty()->SetColor(0.0, 1.0, 0.0);
			cursorActor->GetProperty()->SetOpacity(1.0);

#pragma endregion cursorSphereVtkInicialization

#pragma region cursorCubeVtkInitialization

			cursorRadiusCube = vtkCubeSource::New();
			cursorRadiusCube->SetCenter(0.0, 0.0, 0.0);
			cursorRadiusCube->SetXLength(10.0);
			cursorRadiusCube->SetYLength(10.0);
			cursorRadiusCube->SetZLength(10.0);

			cursorCubeExtractEdges = vtkOutlineFilter::New();
			cursorCubeExtractEdges->SetInput(cursorRadiusCube->GetOutput());

			cursorCubeMapper = vtkPolyDataMapper::New();
			cursorCubeMapper->SetInput(cursorCubeExtractEdges->GetOutput());

			cursorCubeActor = vtkActor::New();
			cursorCubeActor->SetMapper(cursorCubeMapper);
			cursorCubeActor->GetProperty()->SetColor(0.1, 0.1, 0.3);

#pragma endregion cursorCubeVtkInitialization

			sliceData = aggregationFilter->GetOutput();

			sliceMapper = vtkImageMapper::New();
			sliceMapper->SetInput(sliceData);

			sliceActor = vtkActor2D::New();
			sliceActor->SetMapper(sliceMapper);
			sliceActor->SetLayerNumber(0);

			rectangleLine1.SetColor(0.1, 0.1, 0.3);
			rectangleLine2.SetColor(0.1, 0.1, 0.3);
			rectangleLine3.SetColor(0.1, 0.1, 0.3);
			rectangleLine4.SetColor(0.1, 0.1, 0.3);
			rectangleLine4.GetActor()->SetPosition(200.0, 200.0);


			cursorLine1.SetColor(0.0, 1.0, 0.0);
			cursorLine2.SetColor(0.0, 1.0, 0.0);

			sliceRenderer = vtkRenderer::New();
			GetRenderWindow()->AddRenderer(sliceRenderer);
			sliceRenderer->AddActor(sliceActor);
			sliceRenderer->AddActor(rectangleLine1.GetActor());
			sliceRenderer->AddActor(rectangleLine2.GetActor());
			sliceRenderer->AddActor(rectangleLine3.GetActor());
			sliceRenderer->AddActor(rectangleLine4.GetActor());
			sliceRenderer->AddActor(cursorLine1.GetActor());
			sliceRenderer->AddActor(cursorLine2.GetActor());

			std::cout << "Create renderer..." << std::endl; // DEBUG

			_renImageData = vtkOpenGLRenderer::New(); 
			_renImageData->SetUseDepthPeeling(1);  //This and next 2 lines enables depth peeling
			_renImageData->SetMaximumNumberOfPeels(50);
			_renImageData->SetOcclusionRatio(0.2);

			GetRenderWindow()->AddRenderer( _renImageData );

			double leftViewport[4] = {0.0, 0.0, 0.5, 1.0};
			double rightViewport[4] = {0.5, 0.0, 1.0, 1.0};

			_renImageData->SetViewport(leftViewport);
			sliceRenderer->SetViewport(rightViewport);

			std::vector< tissue >::iterator it;
			for (it = tissues.begin(); it != tissues.end(); it++)
			{
				_renImageData->AddActor((*it).GetActor());
			}
			_renImageData->AddActor(cursorActor);
			_renImageData->AddActor(cursorCubeActor);

			vtkRenderWindow *rWin;
			rWin = GetRenderWindow();

			//rWin->SetAlphaBitPlanes(1);  This and next line enables depth peeling
			//rWin->SetMultiSamples(0);

			vtkRenderWindowInteractor *iren;
			iren = GetInteractor();
			iren->SetRenderWindow( rWin );

			_availableSlots.clear();
			_availableSlots.push_back( SETSELECTED );
			_availableSlots.push_back( ZOOM );
			_availableSlots.push_back( MOVE );
			_availableSlots.push_back( ROTATEAXISX );
			_availableSlots.push_back( ROTATEAXISY );
			_availableSlots.push_back( ROTATEAXISZ );
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

		void m4dGUIOGLHapticViewerWidget::slotSetScale(double scale)
		{
			cursor->SetScale(scale);
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

		void m4dGUIOGLHapticViewerWidget::slotMove( int amountH, int amountV ) 
		{
			_renImageData->GetActiveCamera()->SetPosition(((double)amountH)/20.0, ((double)amountV)/20.0, 1.0);
		}

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
				_renImageData->RemoveActor(cursorCubeActor);
				sliceRenderer->RemoveActor(sliceActor);
				sliceRenderer->RemoveActor(rectangleLine1.GetActor());
				sliceRenderer->RemoveActor(rectangleLine2.GetActor());
				sliceRenderer->RemoveActor(rectangleLine3.GetActor());
				sliceRenderer->RemoveActor(rectangleLine4.GetActor());
				sliceRenderer->RemoveActor(cursorLine1.GetActor());
				sliceRenderer->RemoveActor(cursorLine2.GetActor());
				GetRenderWindow()->RemoveRenderer( _renImageData );
				GetRenderWindow()->RemoveRenderer(sliceRenderer);
				sliceRenderer->Delete();
				sliceRenderer = vtkRenderer::New();
				_renImageData->Delete();
				_renImageData = vtkOpenGLRenderer::New();
				_renImageData->SetUseDepthPeeling(1);  //This and next 2 lines enables depth peeling
				_renImageData->SetMaximumNumberOfPeels(50);
				_renImageData->SetOcclusionRatio(0.2);
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
				_renImageData->AddActor(cursorCubeActor);
				sliceRenderer->AddActor(sliceActor);
				sliceRenderer->AddActor(rectangleLine1.GetActor());
				sliceRenderer->AddActor(rectangleLine2.GetActor());
				sliceRenderer->AddActor(rectangleLine3.GetActor());
				sliceRenderer->AddActor(rectangleLine4.GetActor());
				sliceRenderer->AddActor(cursorLine1.GetActor());
				sliceRenderer->AddActor(cursorLine2.GetActor());
				GetRenderWindow()->AddRenderer( _renImageData );
				GetRenderWindow()->AddRenderer(sliceRenderer);
				//if ( _selected ) _renImageData->AddViewProp( _actor2DSelected );
				//_renImageData->AddViewProp( _actor2DPlugged );
				_plugged = true;
				double leftViewport[4] = {0.0, 0.0, 0.5, 1.0};
				double rightViewport[4] = {0.5, 0.0, 1.0, 1.0};

				_renImageData->SetViewport(leftViewport);
				sliceRenderer->SetViewport(rightViewport);
				GetRenderWindow()->Render();
				resetTransitionFunction();
				break;
			}
		}

		void m4dGUIOGLHapticViewerWidget::update()
		{
			reloadCursorParameters();
			GetRenderWindow()->Render();
		}

		void m4dGUIOGLHapticViewerWidget::resetTransitionFunction()
		{
			vtkImageData* input = aggregationFilter->GetOutput();
			int extents[6];
			input->GetExtent(extents);
			unsigned short minVolumeValue = (unsigned short)input->GetScalarTypeMax();
			unsigned short maxVolumeValue = (unsigned short)input->GetScalarTypeMin();
			for (int i = extents[0]; i < extents[1]; ++i)
			{
				for (int j = extents[2]; j < extents[3]; ++j)
				{
					for (int k = extents[4]; k < extents[5]; ++k)
					{
						for (int c = 0; c < input->GetNumberOfScalarComponents(); ++c)
						{
							unsigned short result = (unsigned short)input->GetScalarComponentAsDouble(i,j,k,c);
							if (minVolumeValue > result)
								minVolumeValue = result;
							if (maxVolumeValue < result)
								maxVolumeValue = result;
						}
					}
				}
			}
			hapticForceTransitionFunction->Reset(minVolumeValue, maxVolumeValue, 0.0, 1.0);
			emit hapticForceTransitionFunctionChanged();
		}

		void m4dGUIOGLHapticViewerWidget::slotTransitionFunctionResetDemanded()
		{
			resetTransitionFunction();
		}

		void m4dGUIOGLHapticViewerWidget::resetSliceViewPosition()
		{
			vtkImageData* i = _imageData->GetOutput();
			int extent[6];
			i->GetExtent(extent);
			int moveX = (width() / 2 - (extent[1] + extent[0])) / 2;
			int moveY = (height() - (extent[3] + extent[2])) / 2; 

			sliceActor->SetPosition(moveX, moveY);
			cursorLine1.GetActor()->SetPosition(moveX, moveY);
			cursorLine2.GetActor()->SetPosition(moveX, moveY);
			rectangleLine1.GetActor()->SetPosition(moveX, moveY);
			rectangleLine2.GetActor()->SetPosition(moveX, moveY);
			rectangleLine3.GetActor()->SetPosition(moveX, moveY);
			rectangleLine4.GetActor()->SetPosition(moveX, moveY);
		}

		void m4dGUIOGLHapticViewerWidget::slotZoomInHaptic()
		{
			double scale = cursor->GetScale();
			cursor->SetScale((scale / 3.0) * 2.0);
		}

		void m4dGUIOGLHapticViewerWidget::slotZoomOutHaptic()
		{
			double scale = cursor->GetScale();
			cursor->SetScale((scale / 2.0) * 3.0);
		}
	} /* namespace Viewer */
} /* namespace M4D */
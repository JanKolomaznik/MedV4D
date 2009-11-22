/**
*  @ingroup gui
*  @file m4dGUIOpenGLViewerWidget.cpp
*  @brief some brief
*/

#include "m4dGUIOGLHapticViewerWidget.h"
#include <sstream>
#include <QtGui>

namespace M4D
{
	namespace Viewer
	{
		m4dGUIOGLHapticViewerWidget::m4dGUIOGLHapticViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent) : QGLWidget(parent)
		{
			_index = index;
			_inPort = new Imaging::InputPortTyped<Imaging::AbstractImage>();
			_inputPorts.AppendPort( _inPort );
			setInputPort( conn );
		}
		m4dGUIOGLHapticViewerWidget::m4dGUIOGLHapticViewerWidget(unsigned int index, QWidget *parent) : QGLWidget(parent)
		{
			_index = index;
			_inPort = new Imaging::InputPortTyped<Imaging::AbstractImage>();
			_inputPorts.AppendPort( _inPort );
			setInputPort( );
		}
		m4dGUIOGLHapticViewerWidget::~m4dGUIOGLHapticViewerWidget()
		{
		}
		void m4dGUIOGLHapticViewerWidget::setInputPort()
		{
			_inPort->UnPlug();
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
		{
			if ( !conn )
			{
				setInputPort();
				return;
			}
			conn->ConnectConsumer( *_inPort );
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::setUnSelected()
		{

		}
		void m4dGUIOGLHapticViewerWidget::setSelected()
		{

		}
		m4dGUIOGLHapticViewerWidget::AvailableSlots m4dGUIOGLHapticViewerWidget::getAvailableSlots()
		{
			return _availableSlots;
		}
		QWidget* m4dGUIOGLHapticViewerWidget::operator ()()
		{
			return (QGLWidget*)this;
		}
		void m4dGUIOGLHapticViewerWidget::ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction )
		{
			emit signalMessageHandler( msg->msgID );
		}
		void m4dGUIOGLHapticViewerWidget::mousePressEvent(QMouseEvent *event)
		{
			if (_eventHandler)
			{
				_eventHandler->mousePressEvent(event);
				return;
			}
			_lastMousePosition = QPoint(event->x(), event->y());
			_rightButton = event->buttons() == Qt::RightButton;
			_leftButton = event->buttons() == Qt::LeftButton;
			updateGL();

		}
		void m4dGUIOGLHapticViewerWidget::mouseReleaseEvent(QMouseEvent *event)
		{
			if (_eventHandler)
			{
				_eventHandler->mouseReleaseEvent(event);
				return;
			}
			_rightButton = _rightButton && !(event->buttons() == Qt::RightButton);
			_leftButton = _leftButton && !(event->buttons() == Qt::LeftButton);
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::mouseMoveEvent(QMouseEvent *event)
		{
			if (_eventHandler)
			{
				_eventHandler->mouseMoveEvent(event);
				return;
			}
			if (_leftButton)
			{
				float xMove = ((float)(event->x() - _lastMousePosition.x())) / ((float)(width()));
				float yMove = ((float)(event->y() - _lastMousePosition.y())) / ((float)(height()));

				_rotateY += xMove * 5.0;
				_rotateX += yMove * 5.0; // TODO
			}
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::wheelEvent(QWheelEvent *event)
		{
			if (_eventHandler)
			{
				_eventHandler->wheelEvent(event);
				return;
			}
			int numDegrees = event->delta() / 8;
			int numSteps = numDegrees / 15;

			_zoom += numSteps;

			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::mouseDoubleClickEvent ( QMouseEvent * event )
		{
			if (_eventHandler)
			{
				_eventHandler->mouseDoubleClickEvent(event);
				return;
			}
		}
		void m4dGUIOGLHapticViewerWidget::keyPressEvent ( QKeyEvent * event )
		{
			if (_eventHandler)
			{
				_eventHandler->keyPressEvent(event);
				return;
			}
		}
		void m4dGUIOGLHapticViewerWidget::keyReleaseEvent ( QKeyEvent * event )
		{
			if (_eventHandler)
			{
				_eventHandler->keyReleaseEvent(event);
				return;
			}
		}
		void m4dGUIOGLHapticViewerWidget::DrawTriangle(float x, float y, float z, float size)
		{
			glBegin(GL_TRIANGLES);

			float height = size * sqrt(3.0) / 2;
			float bottom = y - (height / 3);
			float top = bottom + height;

			glVertex3f(x, top, z);
			glVertex3f(x - (size / 2.0f), bottom, z);
			glVertex3f(x + (size / 2.0f), bottom, z);

			glEnd();
		}
		void m4dGUIOGLHapticViewerWidget::paintGL()
		{
			//////////////////////////////////////////////////////////////////////////
			// Testing
			//////////////////////////////////////////////////////////////////////////
			//int x = 3;
			//int y = 2;
			//
			//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// Vymaže obrazovku a hloubkový buffer
			//glClearColor(0.0, 0.0, 0.0, 0.0);
			//glLoadIdentity();// Reset matice								
			//
			////glEnable(GL_POINT_SMOOTH);
			//
			//glPointSize(3.0f);
			//glColor3f(1.0f,1.0f,1.0f);
			//glBegin(GL_POINTS);
			//for (float i = -x; i < x; i+=0.2)
			//{
			//	for (float j = -y; j < y; j+=0.2)
			//	{
			//		for (float k = -x; k < x; k+=0.2)
			//		{
			//			glVertex3f((float) i, (float)j, (float)k);
			//		}
			//	}
			//}
			//glEnd();

			if (! _inPort->IsPlugged() )
			{
				return;
			}

			//GLfloat zOffset = -2.0f;

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// Vymaže obrazovku a hloubkový buffer
			glClearColor(0.0, 0.0, 0.0, 0.0);
			glLoadIdentity();
			
			/*glPointSize(3.0f);
			glColor3f(1.0f,1.0f,.0f);
			glBegin(GL_POINTS);

			float var = 3.0 / ((float)_inPort->GetDatasetTyped().GetDimensionExtents(0).maximum-(float)_inPort->GetDatasetTyped().GetDimensionExtents(0).minimum);
			
			int _imageID = _inPort->GetDatasetTyped().GetElementTypeID();

			int64 result = 0;
			
			if (_inPort->GetDatasetTyped().GetDimension() == 2)
			{
				for (int i = _inPort->GetDatasetTyped().GetDimensionExtents(0).minimum; i < _inPort->GetDatasetTyped().GetDimensionExtents(0).maximum; i++)
				{
					for (int j = _inPort->GetDatasetTyped().GetDimensionExtents(1).minimum; j < _inPort->GetDatasetTyped().GetDimensionExtents(1).maximum; j++)
					{
						NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
							_imageID, result = Imaging::Image< TTYPE, 2 >::CastAbstractImage(_inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >(i, j) ) );
						if (result != 0)
						{
							glColor3f(0.0f + ((GLfloat)result / 256.0),0.0f,0.0f);
							glVertex3f(-1.5+(float)(i*var), -1.5+(float)(j*var), zOffset);
						}
					}
				}
			}
			else
				if (_inPort->GetDatasetTyped().GetDimension() == 3)
				{
					glTranslatef(0.0f, 0.0f, -10.0f);
					glRotatef(45.0f, 1.0f, 1.0f, 0.0f);
					for (int i = _inPort->GetDatasetTyped().GetDimensionExtents(0).minimum; i < _inPort->GetDatasetTyped().GetDimensionExtents(0).maximum; i++)
					{
						for (int j = _inPort->GetDatasetTyped().GetDimensionExtents(1).minimum; j < _inPort->GetDatasetTyped().GetDimensionExtents(1).maximum; j++)
						{
							for (int k = _inPort->GetDatasetTyped().GetDimensionExtents(2).minimum; k < _inPort->GetDatasetTyped().GetDimensionExtents(2).maximum; k++)
							{
								NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
									_imageID, result = Imaging::Image< TTYPE, 3 >::CastAbstractImage(_inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >(i, j, k) ) );
								if (result != 0)
								{
									glColor3f(0.0f + ((GLfloat)result / 32.0),0.0f,0.0f);
									glVertex3f(-1.5+(float)(i*var), -1.5+(float)(j*var), -1.5+(float)(k*var));
								}
							}
						}
					}
					glLoadIdentity();
				}

			glEnd();*/

			float size = ((float)_inPort->GetDatasetTyped().GetDimensionExtents(0).maximum-(float)_inPort->GetDatasetTyped().GetDimensionExtents(0).minimum);
			float var = _imageSize / size;
			int _imageID = _inPort->GetDatasetTyped().GetElementTypeID();
			int64 result = 0;
			
			for (int i = _inPort->GetDatasetTyped().GetDimensionExtents(0).minimum; i < _inPort->GetDatasetTyped().GetDimensionExtents(0).maximum; i++)
			{
				for (int j = _inPort->GetDatasetTyped().GetDimensionExtents(1).minimum; j < _inPort->GetDatasetTyped().GetDimensionExtents(1).maximum; j++)
				{
					for (int k = _inPort->GetDatasetTyped().GetDimensionExtents(2).minimum; k < _inPort->GetDatasetTyped().GetDimensionExtents(2).maximum; k++)
					{
						NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
							_imageID, result = Imaging::Image< TTYPE, 3 >::CastAbstractImage(_inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >(i, j, k) ) );
						if (result != 0)
						{
							glLoadIdentity();
							glTranslatef(0.0f, 0.0f, _zoom);
							glRotatef(_rotateX, 1.0f, 0.0f, 0.0f);
							glRotatef(_rotateY, 0.0f, 1.0f, 0.0f);
							glTranslatef( - _imageSize / 2.0 + i * var,  - _imageSize / 2.0 + j * var,  - _imageSize / 2.0 + k * var);
							glColor3f(0.0f + ((GLfloat)result / 32.0),0.0f,0.0f);
							DrawTriangle(0.0f, 0.0f, 0.0f, var / _imageSize);
						}
					}
				}
			}
			
			//glLoadIdentity();
			//glColor3f(1.0f, 0.0f, 0.0f);
			//glTranslatef(0.0f, 0.0f, -5.0f);
			//DrawTriangle(0.0f, 0.0f, -5.0f, 1.0f);
			//glLoadIdentity();

			glFlush();

		}
		void m4dGUIOGLHapticViewerWidget::resizeGL(int winW, int winH)
		{
			if (winH == 0)
			{
				winH = 1;
			}
			
			glViewport(0, 0, winW, winH);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluPerspective(45.0f,(GLfloat)winW/(GLfloat)winH,0.1f,100.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			updateGL();
		}
		void m4dGUIOGLHapticViewerWidget::initializeGL()
		{
			_imageSize = 10.0;
			_imageHeight = _imageSize;
			_imageWidth = _imageSize;

			_zoom = -_imageSize;
			_rotateY = 0.0;
			_rotateX = 0.0;
			
			glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
			glClearDepth(1.0f);
			glDepthFunc(GL_LEQUAL);
			glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
		}
		void m4dGUIOGLHapticViewerWidget::slotSetButtonHandler( ButtonHandler hnd, MouseButton btn )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotSetSelected(bool selected)
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotSetSliceNum( size_t num )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotSetOneSliceMode()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotToggleFlipHorizontal()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotToggleFlipVertical()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotAddLeftSideData( std::string data )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotAddRightSideData( std::string data )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotClearLeftSideData()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotClearRightSideData()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotTogglePrintData()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotTogglePrintShapeData()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotZoom( int amount )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotMove( int amountH, int amountV )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotAdjustContrastBrightness( int amountB, int amountC )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotNewPoint( double x, double y, double z )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotNewShape( double x, double y, double z )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotDeletePoint()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotDeleteShape()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotDeleteAll()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotRotateAxisX( double x )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotRotateAxisY( double y )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotRotateAxisZ( double z )
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotToggleSliceOrientation()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotColorPicker( double x, double y, double z )
		{

		}
		void m4dGUIOGLHapticViewerWidget::updateViewer()
		{

		}
		void m4dGUIOGLHapticViewerWidget::slotMessageHandler( Imaging::PipelineMsgID msgID )
		{
		}
	}
}
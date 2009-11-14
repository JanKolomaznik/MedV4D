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

			float zOffset = 2.0;

			if (_inPort->GetDatasetTyped().GetDimension() == 2)
			{
				Imaging::Image< TTYPE, 3 >::CastAbstractImage(_inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >( (int)coords[0], (int)coords[1], (int)coords[2] ) )
			}
			else
				if (_inPort->GetDatasetTyped().GetDimension() == 3)
				{
					zOffset = 1.0;
				}

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
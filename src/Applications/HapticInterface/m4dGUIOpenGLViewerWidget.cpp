/**
*  @ingroup gui
*  @file m4dGUIOpenGLViewerWidget.cpp
*  @brief some brief
*/

#include "m4dGUIOpenGLViewerWidget.h"

namespace M4D
{
	namespace Viewer
	{
		m4dGUIOpenGlViewerWidget::m4dGUIOpenGlViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent) : QGLWidget(parent)
		{
			_index = index;
			updateGL();
		}
		m4dGUIOpenGlViewerWidget::m4dGUIOpenGlViewerWidget(unsigned int index, QWidget *parent) : QGLWidget(parent)
		{
			_index = index;
			updateGL();
		}
		m4dGUIOpenGlViewerWidget::~m4dGUIOpenGlViewerWidget()
		{
		}
		void m4dGUIOpenGlViewerWidget::setInputPort()
		{

		}
		void m4dGUIOpenGlViewerWidget::setInputPort( Imaging::ConnectionInterface* conn )
		{

		}
		void m4dGUIOpenGlViewerWidget::setUnSelected()
		{

		}
		void m4dGUIOpenGlViewerWidget::setSelected()
		{

		}
		m4dGUIOpenGlViewerWidget::AvailableSlots m4dGUIOpenGlViewerWidget::getAvailableSlots()
		{
			return _availableSlots;
		}
		QWidget* m4dGUIOpenGlViewerWidget::operator ()()
		{
			return (QGLWidget*)this;
		}
		void m4dGUIOpenGlViewerWidget::ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction )
		{
			emit signalMessageHandler( msg->msgID );
		}
		void m4dGUIOpenGlViewerWidget::paintGL()
		{
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// Vymaže obrazovku a hloubkový buffer
			glClearColor(1.0, 0.8, 0.0, 0.0);
			glLoadIdentity();// Reset matice								
			glTranslatef(-1.5f,0.0f,-6.0f);// Posun doleva a do hloubky
			
			glBegin(GL_TRIANGLES);// Zaèátek kreslení trojúhelníkù
			glColor3f(1.0f, 0.0f, 0.0f);
			glVertex3f( 0.0f, 1.0f, 0.0f);// Horní bod
			glVertex3f(-1.0f,-1.0f, 0.0f);// Levý dolní bod
			glVertex3f( 1.0f,-1.0f, 0.0f);// Pravý dolní bod
			glEnd();// Ukonèení kreslení trojúhelníkù

			glTranslatef(3.0f,0.0f,0.0f);// Posun o 3 jednotky doprava

			glBegin(GL_QUADS);// Zaèátek kreslení obdélníkù
			glVertex3f(-1.0f, 1.0f, 0.0f);// Levý horní bod
			glVertex3f( 1.0f, 1.0f, 0.0f);// Pravý horní bod
			glVertex3f( 1.0f,-1.0f, 0.0f);// Pravý dolní bod
			glVertex3f(-1.0f,-1.0f, 0.0f);// Levý dolní bod
			glEnd();// Konec kreslení obdélníkù

			glFlush();

		}
		void m4dGUIOpenGlViewerWidget::resizeGL(int winW, int winH)
		{
			glViewport(0, 0, width(), height());
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluPerspective(45.0f,(GLfloat)width()/(GLfloat)height(),0.1f,100.0f);
			glMatrixMode(GL_MODELVIEW);
			updateGL();
		}
		void m4dGUIOpenGlViewerWidget::slotSetButtonHandler( ButtonHandler hnd, MouseButton btn )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotSetSelected(bool selected)
		{

		}
		void m4dGUIOpenGlViewerWidget::slotSetSliceNum( size_t num )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotSetOneSliceMode()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotToggleFlipHorizontal()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotToggleFlipVertical()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotAddLeftSideData( std::string data )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotAddRightSideData( std::string data )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotClearLeftSideData()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotClearRightSideData()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotTogglePrintData()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotTogglePrintShapeData()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotZoom( int amount )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotMove( int amountH, int amountV )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotAdjustContrastBrightness( int amountB, int amountC )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotNewPoint( double x, double y, double z )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotNewShape( double x, double y, double z )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotDeletePoint()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotDeleteShape()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotDeleteAll()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotRotateAxisX( double x )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotRotateAxisY( double y )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotRotateAxisZ( double z )
		{

		}
		void m4dGUIOpenGlViewerWidget::slotToggleSliceOrientation()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotColorPicker( double x, double y, double z )
		{

		}
		void m4dGUIOpenGlViewerWidget::updateViewer()
		{

		}
		void m4dGUIOpenGlViewerWidget::slotMessageHandler( Imaging::PipelineMsgID msgID )
		{
		}
	}
}
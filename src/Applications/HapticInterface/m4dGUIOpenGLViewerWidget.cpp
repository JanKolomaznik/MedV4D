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
		}
		m4dGUIOpenGlViewerWidget::m4dGUIOpenGlViewerWidget(unsigned int index, QWidget *parent) : QGLWidget(parent)
		{
			_index = index;
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
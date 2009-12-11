/**
* @ingroup gui 
* @author Jan Bim 
* @file m4dGUIOpenGLViewerWidget.h 
* @{ 
**/

#ifndef M4D_GUI_OPENGLVIEWERWIDGET_H_
#define M4D_GUI_OPENGLVIEWERWIDGET_H_

#define _MSVC
#include <chai3d.h>

#include <QtOpenGL>
#include <list>
#include <string>
#include <map>
#include "Imaging/Imaging.h"
#include "common/Common.h"
#include "..\..\gui\widgets\m4dGUIAbstractViewerWidget.h"

namespace M4D
{
	namespace Viewer
	{
		/**
		* Class that shows anything you describe in openGL.
		*/
		class m4dGUIOGLHapticViewerWidget : public m4dGUIAbstractViewerWidget, public QGLWidget
		{
			Q_OBJECT

		public:

			/**
			* Constructor.
			*  @param index the index of the viewer
			*  @param parent the parent widget of the viewer
			*/
			m4dGUIOGLHapticViewerWidget( unsigned index, QWidget *parent = 0 );

			/**
			* Construtor.
			*  @param conn the connection that connects the viewer
			*  @param index the index of the viewer
			*  @param parent the parent widget of the viewer
			*/
			m4dGUIOGLHapticViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 );

			~m4dGUIOGLHapticViewerWidget();

			/**
			* Disconnects the input port of the viewer.
			*/
			virtual void setInputPort();

			/**
			* Connects the input port of the viewer.
			*  @param conn the connection that connects the viewer
			*/
			virtual void setInputPort( Imaging::ConnectionInterface* conn );

			/**
			* Set the viewer to not selected.
			*/
			virtual void setUnSelected();

			/**
			* Set the viewer to selected.
			*/
			virtual void setSelected();

			/**
			* Find out which viewer slots are implemented in the given viewer.
			*  @return list of integers indicating the implemented viewer slots
			*/
			virtual AvailableSlots getAvailableSlots();

			/**
			* Cast explicitly the viewer to a QWidget. It is necessary for being able to add
			* the widget to other Qt widgets - this class has only a QObject base; the inheriting
			* class has to inherit from QWidget (the reason for this is the problem of multiple
			* inheritence, since the inheriting class will probably inherit from another subclass
			* of QWidget, like QVTKWidget or QGLWidget).
			*  @return Pointer that is casted to the QWidget base of the implementing class
			*/
			virtual QWidget* operator()();

			void DrawTriangle(float x, float y, float z, float size);

			virtual void initializeGL();

			virtual void loadImageParams();

			virtual void ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction );

			virtual void paintGL();

			virtual void resizeGL(int winW, int winH);

			virtual void mousePressEvent(QMouseEvent *event);

			/**
			* Method inherited from QGLWidget. It is called whenever a mouse button is released
			* above the widget.
			*  @param event the mouse release event to be handled
			*/
			virtual void mouseReleaseEvent(QMouseEvent *event);

			/**
			* Method inherited from QGLWidget. It is called whenever the mouse is moved above a widget.
			*  @param event the mouse move event to be handled
			*/
			virtual void mouseMoveEvent(QMouseEvent *event);

			/**
			* Method inherited from QGLWidget. It is called whenever the wheel is moved above the widget.
			*  @param event the wheel event to be handled.
			*/
			virtual void wheelEvent(QWheelEvent *event);

			/**
			* Method inherited from QGLWidget. It is called whenever a mouse button is double-clicked
			* above the widget.
			*  @param event the mouse double-click event to be handled
			*/
			virtual void mouseDoubleClickEvent ( QMouseEvent * event );

			/**
			* Method inherited from QGLWidget. It is called whenever a keyboard key is pressed
			* above the widget.
			*  @param event the key press event to be handled
			*/
			virtual void keyPressEvent ( QKeyEvent * event );

			/**
			* Method inherited from QGLWidget. It is called whenever a keyboard key is released
			* above the widget.
			*  @param event the key release event to be handled
			*/
			virtual void keyReleaseEvent ( QKeyEvent * event );

			AvailableSlots				_availableSlots;

			virtual void updateViewer();

			public slots:
				/**
				* Slot to connect a given button to a given handler method.
				*  @param hnd the handler method
				*  @param btn the button to connect to the method
				*/
				virtual void slotSetButtonHandler( ButtonHandler hnd, MouseButton btn );

				/**
				* Slot to set if the viewer is selected or not.
				*  @param selected tells if the viewer should be selected or not
				*/
				virtual void slotSetSelected( bool selected );

				/**
				* Slot to set the current slice number.
				*  @param num the slice number to be set
				*/
				virtual void slotSetSliceNum( size_t num );

				/**
				* Slot to set the viewer to show one slice at once.
				*/
				virtual void slotSetOneSliceMode();

				/**
				* Slot to set the viewer to show several slices at once.
				*  @param slicesPerRow how many slices will be shown in one row
				*  @param slicesPerColumn how many slices will be shown in one column
				*/
				virtual void slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn );

				/**
				* Slot to toggle vertical flip.
				*/
				virtual void slotToggleFlipVertical();

				/**
				* Slot to toggle horizontal flip.
				*/
				virtual void slotToggleFlipHorizontal();

				/**
				* Slot to add some text data to show on the left side of the viewer.
				*  @param data the value of the given data
				*/
				virtual void slotAddLeftSideData( std::string data );

				/**
				* Slot to add some text data to show on the right side of the viewer.
				*  @param data the value of the given data
				*/
				virtual void slotAddRightSideData( std::string data );

				/**
				* Slot to clear all data from the left side of the viewer.
				*/
				virtual void slotClearLeftSideData();

				/**
				* Slot to clear all data from the right side of the viewer.
				*/
				virtual void slotClearRightSideData();

				/**
				* Slot to toggle the printing of data on the viewer.
				*/
				virtual void slotTogglePrintData();

				/**
				* Slot to toggle the printing of the selected shapes' information on the viewer.
				*/
				virtual void slotTogglePrintShapeData();

				/**
				* Slot to zoom the image.
				*  @param amount how much we want to zoom. Positive value means zoom in,
				*                negative value means zoom out.
				*/
				virtual void slotZoom( int amount );

				/**
				* Slot to move the image.
				*  @param amountH the amount to move the image horizontally
				*  @param amountV the amount to move the image vertically
				*/
				virtual void slotMove( int amountH, int amountV );

				/**
				* Slot to adjust the brightness and contrast of the image.
				*  @param amountB the amount to adjust the brightness
				*  @param amountC the amount to adjust the contrast
				*/
				virtual void slotAdjustContrastBrightness( int amountB, int amountC );

				/**
				* Slot to add a new point to the last created shape of the list of selected
				* shapes.
				*  @param x the x coordinate of the point
				*  @param y the y coordinate of the point
				*  @param z the z coordinate of the point
				*/
				virtual void slotNewPoint( double x, double y, double z );

				/**
				* Slot to add a new shape to the list of selected shapes.
				*  @param x the x coordinate of the point
				*  @param y the y coordinate of the point
				*  @param z the z coordinate of the point
				*/
				virtual void slotNewShape( double x, double y, double z );

				/**
				* Slot to delete the last selected point.
				*/
				virtual void slotDeletePoint();

				/**
				* Slot to delete the last selected shape.
				*/
				virtual void slotDeleteShape();

				/**
				* Slot to erase all selected shapes and points.
				*/
				virtual void slotDeleteAll();

				/**
				* Slot to rotate the scene around the x axis.
				*  @param x the angle that the scene is to be rotated by
				*/
				virtual void slotRotateAxisX( double x );

				/**
				* Slot to rotate the scene around the y axis.
				*  @param y the angle that the scene is to be rotated by
				*/
				virtual void slotRotateAxisY( double y );

				/**
				* Slot to rotate the scene around the z axis.
				*  @param z the angle that the scene is to be rotated by
				*/
				virtual void slotRotateAxisZ( double z );

				/**
				* Slot to toggle the orientation of the slice viewing axes.
				* xy -> yz -> zx
				*/
				virtual void slotToggleSliceOrientation();

				/**
				* Slot to pick the color of the pixel at the given position.
				*  @param x the x coordinate
				*  @param y the y coordinate
				*  @param z the z coordinate
				*/
				virtual void slotColorPicker( double x, double y, double z );

			protected:

				QPoint _lastMousePosition;
				bool _leftButton;
				bool _rightButton;
				float _imageSize;
				float _imageWidth;
				float _imageHeight;
				float _zoom;
				float _rotateX, _rotateZ, _rotateY;
				int _imageID;
				float _sizeX, _sizeY, _sizeZ;
				float _minX, _minY, _minZ;
				float _varX, _varY, _varZ;
				float _trianglSize;
				int64 _minValue, _maxValue;

				/**
				* The input port that can be connected to the pipeline.
				*/
				Imaging::InputPortTyped< Imaging::AImage >	*_inPort;


				// a haptic device handler
				cHapticDeviceHandler* handler;

				// a pointer to a haptic device
				cGenericHapticDevice* hapticDevice;

				// haptic device info
				cHapticDeviceInfo info;

				// number of haptic devices
				int numHapticDevices;

				// last position of haptic device
				cVector3d position;

				// button status of the haptic device
				bool buttonStatus;

				void initializeHaptics();

			protected slots:

				/**
				* Slot to handle incoming message from Image pipeline.
				*  @param msgID the ID of the message
				*/
				virtual void slotMessageHandler( Imaging::PipelineMsgID msgID );	
	

		};

	} /*namespace Viewer*/
} /*namespace M4D*/

#endif

/** @} */


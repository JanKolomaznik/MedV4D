/**
* @ingroup gui 
* @author Jan Bim 
* @file m4dGUIOpenGLViewerWidget.h 
* @{ 
**/

#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_

//#define _MSVC
#include <QWidget>
#include <QVTKWidget.h>
#include <vector>

// VTK includes
#include "vtkOpenGLRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkActor2D.h"
#include "vtkPolyDataMapper2D.h"
#include "vtkProperty2D.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkCamera.h"
#include "vtkImageCast.h"
#include "vtkMarchingCubes.h"
#include "vtkPolyDataNormals.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkSphereSource.h"
#include "vtkCubeSource.h"
#include "vtkOutlineFilter.h"
#include "vtkTubeFilter.h"
#include "vtkSphereSource.h"
#include "vtkCallbackCommand.h"
#include "aggregationFilterForVTK.h"
#include "vtkCommand.h"
#include "vtkImageMapper.h"
#include "vtkLineSource.h"
#include "vtkCornerAnnotation.h"

#include "vtkIntegration/m4dImageDataSource.h"

#include "common/Common.h"
#include "Imaging/Imaging.h"
#include "gui\widgets\m4dGUIAbstractViewerWidget.h"
#include "GUI/widgets/utils/ViewerFactory.h"
#include "hapticCursor.h"
#include "settingsBoxWidget.h"

namespace M4D
{
	namespace Viewer
	{

		/**
		* Class that uses the VTK toolkit to visualize image datasets in 3D.
		*/
		class m4dGUIOGLHapticViewerWidget: public m4dGUIAbstractViewerWidget, public QVTKWidget
		{
			Q_OBJECT

		public:

			struct tissue
			{
			public:
				tissue(vtkAlgorithmOutput* data, double value, double r, double g, double b, double opacity);
				vtkActor* GetActor();
				void deleteInnerItems();
			private:
				vtkMarchingCubes* iso;
				vtkPolyDataNormals* isoNormals;
				vtkPolyDataMapper* isoMapper;
				vtkActor* isoActor;
			};

			struct line
			{
			public:
				line();
				~line();
				void SetPoints(double a_point0[3], double a_point1[3]);
				void SetColor(double a_red, double a_green, double a_blue);
				vtkActor2D* GetActor();
			private:
				vtkLineSource* m_source;
				vtkPolyDataMapper2D* m_mapper;
				vtkActor2D* m_actor;
			};

			/**
			* Constructor.
			*  @param conn the connection that connects the viewer
			*  @param index the index of the viewer
			*  @param parent the parent widget of the viewer
			*/
			m4dGUIOGLHapticViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 );

			/**
			* Constructor.
			*  @param index the index of the viewer
			*  @param parent the parent widget of the viewer
			*/
			m4dGUIOGLHapticViewerWidget( unsigned index, QWidget *parent = 0 );

			/**
			* Destructor.
			*/
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
			* the widget to other Qt widgets through the m4dGUIAbstractViewer interface.
			*  @return Pointer that is casted to the QWidget base of the implementing class
			*/
			virtual QWidget* operator()();


			/**
			* Method for receiving messages - called by sender. ( Implementing from MessageReceiverInterface ).
			* @param msg Smart pointer to message object - we don't have to worry about deallocation
			* @param sendStyle How treat incoming message
			* @param direction defines direction
			*/
			virtual void ReceiveMessage( 
				Imaging::PipelineMessage::Ptr msg,
				Imaging::PipelineMessage::MessageSendStyle sendStyle, 
				Imaging::FlowDirection direction );


			public slots:

				void update();
				/*
				* Qt testing slot
				*/
				virtual void slotSetScale(double scale);

				/**
				* Slot to connect a given button to a given handler method
				* (it has no function in this type of viewer).
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
				* Slot to set the current slice number
				* (it has no function in this type of viewer).
				*  @param num the slice number to be set
				*/
				virtual void slotSetSliceNum( size_t num );

				/**
				* Slot to set the viewer to show one slice at once
				* (it has no function in this type of viewer).
				*/
				virtual void slotSetOneSliceMode();

				/**
				* Slot to set the viewer to show several slices at once
				* (it has no function in this type of viewer).
				*  @param slicesPerRow how many slices will be shown in one row
				*  @param slicesPerColumn how many slices will be shown in one column
				*/
				virtual void slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn );

				/**
				* Slot to toggle vertical flip
				* (it has no function in this type of viewer).
				*/
				virtual void slotToggleFlipVertical();

				/**
				* Slot to toggle horizontal flip
				* (it has no function in this type of viewer).
				*/
				virtual void slotToggleFlipHorizontal();

				/**
				* Slot to add some text data to show on the left side of the viewer
				* (it has no function in this type of viewer).
				*  @param data the value of the given data
				*/
				virtual void slotAddLeftSideData( std::string data );

				/**
				* Slot to add some text data to show on the right side of the viewer
				* (it has no function in this type of viewer).
				*  @param data the value of the given data
				*/
				virtual void slotAddRightSideData( std::string data );

				/**
				* Slot to clear all data from the left side of the viewer
				* (it has no function in this type of viewer).
				*/
				virtual void slotClearLeftSideData();

				/**
				* Slot to clear all data from the right side of the viewer
				* (it has no function in this type of viewer).
				*/
				virtual void slotClearRightSideData();

				/**
				* Slot to toggle the printing of data on the viewer
				* (it has no function in this type of viewer).
				*/
				virtual void slotTogglePrintData();

				/**
				* Slot to toggle the printing of the selected shapes' information on the viewer
				* (it has no function in this type of viewer).
				*/
				virtual void slotTogglePrintShapeData();

				/**
				* Slot to zoom the image
				*  @param amount how much we want to zoom. Positive value means zoom in,
				*                negative value means zoom out.
				*/
				virtual void slotZoom( int amount );

				/**
				* Slot to move the image
				* (it has no function in this type of viewer).
				*  @param amountH the amount to move the image horizontally
				*  @param amountV the amount to move the image vertically
				*/
				virtual void slotMove( int amountH, int amountV );

				/**
				* Slot to adjust the brightness and contrast of the image
				* (it has no function in this type of viewer).
				*  @param amountB the amount to adjust the brightness
				*  @param amountC the amount to adjust the contrast
				*/
				virtual void slotAdjustContrastBrightness( int amountB, int amountC );

				/**
				* Slot to add a new point to the last created shape of the list of selected
				* shapes
				* (it has no function in this type of viewer).
				*  @param x the x coordinate of the point
				*  @param y the y coordinate of the point
				*  @param z the z coordinate of the point
				*/
				virtual void slotNewPoint( double x, double y, double z );

				/**
				* Slot to add a new shape to the list of selected shapes
				* (it has no function in this type of viewer).
				*  @param x the x coordinate of the point
				*  @param y the y coordinate of the point
				*  @param z the z coordinate of the point
				*/
				virtual void slotNewShape( double x, double y, double z );

				/**
				* Slot to delete the last selected point
				* (it has no function in this type of viewer).
				*/
				virtual void slotDeletePoint();

				/**
				* Slot to delete the last selected shape
				* (it has no function in this type of viewer).
				*/
				virtual void slotDeleteShape();

				/**
				* Slot to erase all selected shapes and poitns
				* (it has no function in this type of viewer).
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
				* Slot to toggle the orientation of the slice viewing axes
				* xy -> yz -> zx
				* (it has no function in this type of viewer).
				*/
				virtual void slotToggleSliceOrientation();

				/**
				* Slot to pick the color of the pixel at the given position
				* (it has no function in this type of viewer).
				*  @param x the x coordinate
				*  @param y the y coordinate
				*  @param z the z coordinate
				*/
				virtual void slotColorPicker( double x, double y, double z );

				void updateViewer()
				{ /*TODO*/ }

		signals:
				void hapticForceTransitionFunctionChanged();
				
				protected slots:

					/**
					* Slot to handle incoming message from Image pipeline.
					*  @param msgID the ID of the message
					*/
					virtual void slotMessageHandler( Imaging::PipelineMsgID msgID );

					virtual void slotTransitionFunctionResetDemanded();

					virtual void slotZoomInHaptic();

					virtual void slotZoomOutHaptic();

		protected:

			/**
			* Method inherited from QVTKWidget. It is called whenever the widget is resized.
			*  @param event the resize event to be handled
			*/
			virtual void resizeEvent( QResizeEvent* event );

			/**
			* Method inherited from QVTKWidget. It is called whenever a mouse button is pressed
			* above the widget.
			*  @param event the mouse press event to be handled
			*/
			virtual void mousePressEvent(QMouseEvent *event);

			/**
			* Method inherited from QVTKWidget. It is called whenever a mouse button is double clicked
			* above the widget.
			*  @param event the mouse double-click event to be handled
			*/
			virtual void mouseDoubleClickEvent ( QMouseEvent * event );

			/**
			* Method inherited from QVTKWidget. It is called whenever a mouse cursor is moved
			* above the widget.
			*  @param event the mouse move event to be handled
			*/
			virtual void mouseMoveEvent ( QMouseEvent * event );

			/**
			* Method inherited from QVTKWidget. It is called whenever a mouse button is released
			* above the widget.
			*  @param event the mouse release event to be handled
			*/
			virtual void mouseReleaseEvent ( QMouseEvent * event );

			/**
			* Method inherited from QVTKWidget. It is called whenever a mouse wheel is rolled
			* above the widget.
			*  @param event the wheel event to be handled
			*/
			virtual void wheelEvent ( QWheelEvent * event );

			/**
			* Method inherited from QVTKWidget. It is called whenever a keyboard key is pressed
			* above the widget.
			*  @param event the key press event to be handled
			*/
			virtual void keyPressEvent ( QKeyEvent * event );

			/**
			* Method inherited from QVTKWidget. It is called whenever a keyboard key is released
			* above the widget.
			*  @param event the key release event to be handled
			*/
			virtual void keyReleaseEvent ( QKeyEvent * event );

		private:

			/**
			* Sets the border points appropriately.
			*  @param points the points to be set
			*  @param cells the cells that the points need to be added to
			*  @param pos the distance between the edge of the widget and the given border
			*/
			void setBorderPoints( vtkPoints* points, vtkCellArray *cells, unsigned pos );

			/**
			* Sets the parameters that are dependent on the given image.
			*/
			void setParameters();

			void reloadCursorParameters();

			void resetTransitionFunction();

			void resetSliceViewPosition();

			static void cursorSphereModifiedCallback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData);

			/**
			* True if the viewer is plugged, false otherwise.
			*/
			bool					_plugged;


			/**
			* The input port that can be connected to the pipeline.
			*/
			Imaging::InputPortTyped< Imaging::AImage >	*_inPort;

			/**
			* The object that converts m4d image to vtk image.
			*/
			vtkIntegration::m4dImageDataSource*		_imageData;

			/**
			* VTK image convertor for actually converting images.
			*/
			vtkImageCast*				_iCast;
			
			/*
			* VTK filter for value aggregation
			*/
			aggregationFilterForVtk* aggregationFilter;

			/**
			* 2D actor for "selected" border.
			*/
			vtkActor2D*					_actor2DSelected;

			/**
			* Points of the "selected" border.
			*/
			vtkPoints*					_pointsSelected;

			/**
			* Polygon to help mapping the points of the "selected" border into the scene.
			*/
			vtkPolyData*				_pointsDataSelected;

			/**
			* Mapper that maps the "selected" border into the scene.
			*/
			vtkPolyDataMapper2D*			_pointsDataMapperSelected;

			/**
			* Cell array to hold the "selected" points.
			*/
			vtkCellArray*				_cellsSelected;

			/**
			* 2D actor for "plugged" border.
			*/
			vtkActor2D*					_actor2DPlugged;

			/**
			* Points of the "plugged" border.
			*/
			vtkPoints*					_pointsPlugged;

			/**
			* Polygon to help mapping the points of the "plugged" border into the scene.
			*/
			vtkPolyData*				_pointsDataPlugged;

			/**
			* Mapper that maps the "plugged" border into the scene.
			*/
			vtkPolyDataMapper2D*			_pointsDataMapperPlugged;

			/**
			* Cell array to hold the "plugged" points.
			*/
			vtkCellArray*				_cellsPlugged;

			/*
			* Vector of tissue struct which holds data for each tissue to render
			*/
			std::vector< tissue > tissues;

			/**
			* Renderer to render the whole scene.
			*/
			vtkOpenGLRenderer*			_renImageData;

			/**
			* List of integers indicating which slots are implemented in this type of viewer.
			*/
			AvailableSlots				_availableSlots;

			/*
			* Mapper for cursor sphere
			*/
			vtkPolyDataMapper* cursorMapper;

			/*
			* filter for extracting edges from cube model
			*/
			vtkOutlineFilter* cursorCubeExtractEdges;

			/*
			* Mapper for cursor radius cube
			*/
			vtkPolyDataMapper* cursorCubeMapper;

			/*
			* Actor for cursor sphere
			*/
			vtkActor* cursorActor;

			/*
			* Actor for cursor radius cube
			*/
			vtkActor* cursorCubeActor;

			/*
			* Cursor
			*/
			hapticCursor* cursor;

			/*
			* Cursor graphics source
			*/
			vtkSphereSource* cursorSource;

			/*
			* Cursor radius cube source
			*/
			vtkCubeSource* cursorRadiusCube;

			transitionFunction* hapticForceTransitionFunction;

			SettingsBoxWidget* settings;

			vtkImageData* sliceData;

			vtkImageMapper* sliceMapper;

			vtkActor2D* sliceActor;

			vtkRenderer* sliceRenderer;

			vtkCornerAnnotation* cornerAnnotation;

			line rectangleLine1;
			line rectangleLine2;
			line rectangleLine3;
			line rectangleLine4;

			line cursorLine1;
			line cursorLine2;
		};

		typedef M4D::GUI::GenericViewerFactory< M4D::Viewer::m4dGUIOGLHapticViewerWidget > OGLHapticViewerFactory;

	} /* namespace Viewer */
} /* namespace M4D */

#endif

/** @} */


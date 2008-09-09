/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUISliceViewerWidget.h 
 * @{ 
 **/

#ifndef _M4DGUISLICEVIEWERWIDGET_H
#define _M4DGUISLICEVIEWERWIDGET_H

#include <QtOpenGL>
#include <list>
#include <string>
#include <map>
#include "Imaging/Image.h"
#include "Common.h"
#include "ExceptionBase.h"
#include "Imaging/Ports.h"
#include "Imaging/ImageConnection.h"
#include "GUI/m4dSelection.h"
#include "GUI/m4dGUIAbstractViewerWidget.h"

#define RW 0.3086
#define GW 0.6094
#define BW 0.0820

namespace M4D
{
namespace Viewer
{

/**
 * Class that shows the image dataset slice-by-slice.
 */
class m4dGUISliceViewerWidget : public m4dGUIAbstractViewerWidget, public QGLWidget
{
    Q_OBJECT

public:
    
    /**
     * Type of method for handling mouse movement events.
     */
    typedef void (M4D::Viewer::m4dGUISliceViewerWidget::*ButtonMethods)( int amount1, int amount2 );

    /**
     * Type of method for handling mouse press events.
     */
    typedef void (M4D::Viewer::m4dGUISliceViewerWidget::*SelectMethods)( double x, double y, double z );
    
    /**
     * Constructor.
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    m4dGUISliceViewerWidget( unsigned index, QWidget *parent = 0 );

    /**
     * Construtor.
     *  @param conn the connection that connects the viewer
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    m4dGUISliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 );

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
     * @param sendStyle How treat incoming message of sending message
     * @param direction defines direction
     */
    virtual void ReceiveMessage( 
      Imaging::PipelineMessage::Ptr msg,
      Imaging::PipelineMessage::MessageSendStyle sendStyle,
      Imaging::FlowDirection direction );

    
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
     * Slot to erase all selected shapes and poitns.
     */
    virtual void slotDeleteAll();

    /**
     * Slot to rotate the scene around the x axis
     * (it has no function in this type of viewer).
     *  @param x the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisX( double x );

    /**
     * Slot to rotate the scene around the y axis
     * (it has no function in this type of viewer).
     *  @param y the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisY( double y );

    /**
     * Slot to rotate the scene around the z axis
     * (it has no function in this type of viewer).
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

protected slots:
    
    /**
     * Slot to handle incoming message from Image pipeline.
     *  @param msgID the ID of the message
     */
    virtual void slotMessageHandler( Imaging::PipelineMsgID msgID );

protected:

    /**
     * Checks if a given point is out of bounds of the image.
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     */
    bool checkOutOfBounds( double x, double y );

    /**
     * Check if flips are turned on, and modify point coordinate accordingly.
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     */
    void resolveFlips( double& x, double& y );

    /**
     * Connect a button handler method to a given mouse button.
     *  @param hnd the button handler method to connect
     *  @param btn the mouse button to connect to
     */
    void setButtonHandler( ButtonHandler hnd, MouseButton btn );

    /**
     * Calculates the image pixel position from a picked display position
     * and applies a select method to the found pixel position.
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     *  @param f the select method to be applied to the result
     */
    void ImagePositionSelectionCaller( int x, int y, SelectMethods f );

    /**
     * Set the viewer to display one slice at a time.
     */
    void setOneSliceMode();

    /**
     * Set the viewer to display several slices at a time.
     *  @param slicesPerRow how many slices should be displayed in a row
     *  @param slicesPerColumn how many slices should be displayed in a column
     */
    void setMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn );

    /**
     * Switch the viewer to another slice.
     *  @param dummy does nothing, only needed for the method to be a valid button method
     *  @param amount the amount by which the present slice number should be modified
     */
    void switchSlice( int dummy, int amount );

    /**
     * Toggle horizontal flip.
     */
    void toggleFlipHorizontal();

    /**
     * Toggle vertical flip.
     */
    void toggleFlipVertical();

    /**
     * Add text data to the left side of the viewer.
     *  @param data the text data itself
     */
    void addLeftSideData( std::string data );

    /**
     * Add text data to the right side of the viewer.
     *  @param data the text data itself
     */
    void addRightSideData( std::string data );

    /**
     * Erases all the text data from the left side of the viewer.
     */
    void clearLeftSideData();

    /**
     * Erases all the text data from the right side of the viewer.
     */
    void clearRightSideData();

    /**
     * Toggles text data printing on the screen.
     */
    void togglePrintData();

    /**
     * Method inherited from QGLWidget. It is called whenever the widget is
     * to be updated or updateGL() is called.
     */
    void paintGL();

    /**
     * Method inherited from QGLWidget. It is called whenever the widget is resized.
     *  @param winW the new width of the widget
     *  @param winH the new height of the widget
     */
    void resizeGL(int winW, int winH);

    /**
     * Method inherited from QGLWidget. It is called whenever a mouse button is pressed
     * above the widget.
     *  @param event the mouse press event to be handled
     */
    void mousePressEvent(QMouseEvent *event);

    /**
     * Method inherited from QGLWidget. It is called whenever a mouse button is released
     * above the widget.
     *  @param event the mouse release event to be handled
     */
    void mouseReleaseEvent(QMouseEvent *event);

    /**
     * Method inherited from QGLWidget. It is called whenever the mouse is moved above a widget.
     *  @param event the mouse move event to be handled
     */
    void mouseMoveEvent(QMouseEvent *event);

    /**
     * Method inherited from QGLWidget. It is called whenever the wheel is moved above the widget.
     *  @param event the wheel event to be handled.
     */
    void wheelEvent(QWheelEvent *event);

    /**
     * Sets the current slice number.
     *  @param num the new slice number
     */
    void setSliceNum( size_t num );

    /**
     * Zooms in/out the image.
     *  @param dummy does nothing, only needed for the method to be a valid button method
     *  @param amount the amount by wich the zoom rate should be changed
     */
    void zoomImage( int dummy, int amount );

    /**
     * Moves image.
     *  @param amountH the amount by which the image is to be moved horizontally
     *  @param amountV the amount by which the image is to be moved vertically
     */
    void moveImage( int amountH, int amountV );

    /**
     * Adjusts the contrast and the brightness of the image.
     *  @param amountB the amount by which the brightness is to be adjusted
     *  @param amountC the amount by which the contrast is to be adjusted
     */
    void adjustContrastBrightness( int amountB, int amountC );

    /**
     * Selects a point in the image.
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     *  @param z the z coordinate of the point
     */
    void newPoint( double x, double y, double z );

    /**
     * Starts selecting a new shape in the image.
     *  @param x the x coordinate of the first point of the shape
     *  @param y the y coordinate of the first point of the shape
     *  @param z the z coordinate of the first point of the shape
     */
    void newShape( double x, double y, double z );

    /**
     * Erases the last selected point.
     */
    void deletePoint();

    /**
     * Erases the last selected shape.
     */
    void deleteShape();

    /**
     * Erases all the selected points and shapes.
     */
    void deleteAll();

    /**
     * Pick the color of a voxel.
     *  @param x the x coordinate of the voxel
     *  @param y the y coordinate of the voxel
     *  @param z the z coordinate of the voxel
     */
    void colorPicker( double x, double y, double z );

private:

    /**
     * Sets the parameters that are dependent on the given image.
     */
    void setParameters();

    /**
     * Resets all the parameters to their starting state.
     */
    void resetParameters();

    /**
     * Draws a border rectangle.
     *  @param red the amount of red in the border color
     *  @param green the amount of green in the border color
     *  @param blue the amount of blue in the border color
     *  @param pos the distance between the edge of the widget and the given border
     */
    void borderDrawer( GLfloat red, GLfloat green, GLfloat blue, unsigned pos );

    /**
     * Draws a border indicating that the viewer is plugged into a pipeline.
     */
    void drawPluggedBorder();

    /**
     * Draws a border indicating that the viewer is in selection mode.
     */
    void drawSelectionModeBorder();

    /**
     * Draws a border indicating that the viewer is selected.
     */
    void drawSelectedBorder();

    /**
     * Draws text data.
     *  @param zoomRate the zoom rate that is applied to the image under the text data
     *  @param offset the offset of the drawing
     *  @param sliceNum the number of the slice that is to be drawn
     */
    void drawData( double zoomRate, QPoint offset, int sliceNum );

    /**
     * Draws a slice.
     *  @param sliceNum the number of the slice that is to be drawn
     *  @param zoomRate the zoom rate that is to be applied to the image
     *  @param offset the offset of the image on the viewer
     */
    void drawSlice( int sliceNum, double zoomRate, QPoint offset );

    /**
     * Draws a selected shape.
     *  @param s the shape to be drawn
     *  @param last true if this is the last selected shape, false otherwise
     *  @param sliceNum the number of the slice that the shape is to be drawn on
     *  @param zoomRate the zoom rate that is applied to the shape
     */
    void drawShape( Selection::m4dShape<double>& s, bool last, int sliceNum, double zoomRate );

    /**
     * Prints out the value of the picked voxel at the picked position
     */
    void drawPicked();

    /**
     * Calculate the real width and the real height of the image
     *  @param w reference where the real width will be stored
     *  @param h reference where the real height will be stored
     */
    void calculateWidthHeight( double& w, double& h );

    /**
     * Calculate the optimal zoom rate so that the image would be shown as large as possible
     * while still fitting into the viewer widget.
     */
    void calculateOptimalZoomRate();

    /**
     * Draw text on the screen.
     *  @param xpos the x coordinate of the beginning of the text
     *  @param ypos the y coordinate of the beginning of the text
     *  @param text the text itself
     */
    void textDrawer( int xpos, int ypos, const char* text );

    
    /**
     * The input port that can be connected to the pipeline.
     */
    Imaging::InputPortAbstractImage*		_inPort;

    
    /**
     * The list of selected shapes
     */
    std::list< Selection::m4dShape<double> >	_shapes;

    
    /**
     * < 0 value indicates horizontal flip.
     */
    short					_flipH;

    /**
     * < 0 value indicates vertical flip.
     */
    short					_flipV;

    /**
     * The minimum x, y, z coordinates of the image.
     */
    size_t					_minimum[3];

    /**
     * The maximum x, y, z coordinates of the image.
     */
    size_t					_maximum[3];

    /**
     * The x, y, z axis extents of the image.
     */
    float					_extents[3];

    /**
     * The ID of the voxel element type of the image.
     */
    int						_imageID;

    
    /**
     * True if the image is ready for drawing, false otherwise.
     */
    bool					_ready;

    /**
     * True if the data text of the image is to be drawn, false otherwise.
     */
    bool					_printData;

    /**
     * True if the data text of the shapes is to be drawn, false otherwise.
     */
    bool					_printShapeData;

    /**
     * True if one slice is to be shown at a time, false otherwise.
     */
    bool					_oneSliceMode;

    /**
     * Coordinates of the last clicked position on the viewer.
     */
    QPoint					_lastPos;

    /**
     * The offset of the image on the viewer in one slice mode.
     */
    QPoint					_offset;

    /**
     * The number of the currently viewed slice.
     */
    int						_sliceNum;

    /**
     * How many slices are displayed in a row in more slice mode.
     */
    unsigned					_slicesPerRow;

    /**
     * How many slices are displayed in a column in more slice mode.
     */
    unsigned					_slicesPerColumn;

    /**
     * The rate of zoom of the image on display.
     */
    double					_zoomRate;

    /**
     * The rate of brightness adjustment of the image on display.
     */
    GLint					_brightnessRate;

    /**
     * The rate of contrast adjustment of the image on display.
     */
    GLint					_contrastRate;

    /**
     * The methods to be used when mouse button is pressed while not in selection mode.
     */
    ButtonMethods				_buttonMethods[2];

    /**
     * The methods to be used when mouse button is pressed while in selection mode.
     */
    SelectMethods				_selectMethods[2];

    /**
     * Selection mode enabled/disabled for left and right mouse buttons.
     */
    bool					_selectionMode[2];

    /**
     * List of integers indicating which slots are implemented in this type of viewer.
     */
    AvailableSlots				_availableSlots;

    /**
     * The current slice vieweing orientation - xy, yz, zx.
     */
    SliceOrientation				_sliceOrientation;

    /**
     * True, if a color has been picked recently and needs to be drawn.
     */
    bool					_colorPicker;

    /**
     * The value of the recently picked voxel's color.
     */
    int64					_colorPicked;

    /**
     * The position on display of the recently picked voxel.
     */
    QPoint					_pickedPosition;

    /**
     * The number of the slice where the recent voxel pick has been made.
     */
    int						_slicePicked;

    /**
     * How many dimensions the currently shown image has.
     */
    unsigned					_dimension;
};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

/** @} */


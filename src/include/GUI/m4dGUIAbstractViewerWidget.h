#ifndef _M4DGUIABSTRACTVIEWERWIDGET_H
#define _M4DGUIABSTRACTVIEWERWIDGET_H

#include "Common.h"
#include "Imaging/PipelineMessages.h"
#include "Imaging/Ports.h"
#include <QtGui>

/* defines that indicate the capabilities of the given type of viewer widget */
#define SETBUTTONHANDLER	0
#define SETSELECTED		1
#define SETSLICENUM		2
#define SETONESLICEMODE		3
#define SETMORESLICEMODE	4
#define TOGGLEFLIPVERTICAL	5
#define TOGGLEFLIPHORIZONTAL	6
#define ADDLEFTSIDEDATA		7
#define ADDRIGHTSIDEDATA	8
#define ERASELEFTSIDEDATA	9
#define ERASERIGHTSIDEDATA	10
#define CLEARLEFTSIDEDATA	11
#define CLEARRIGHTSIDEDATA	12
#define TOGGLEPRINTDATA		13
#define TOGGLEPRINTSHAPEDATA	14
#define ZOOM			15
#define MOVE			16
#define CONTRASTBRIGHTNESS	17
#define NEWPOINT		18
#define NEWSHAPE		19
#define DELETEPOINT		20
#define DELETESHAPE		21
#define DELETEALL		22
#define ROTATEAXISX		23
#define ROTATEAXISY		24
#define ROTATEAXISZ		25
#define SETSLICEORIENTATION	26
#define COLORPICKER		27

namespace M4D
{
namespace Viewer
{

/** 
 * Abstract base class of the ViewerWidgets.
 */
class m4dGUIAbstractViewerWidget : public QObject, public Imaging::MessageReceiverInterface
{
    Q_OBJECT

public:

    /**
     * Enumeration to pass to the button handler setting method to know which method to use.
     */
    typedef enum { zoomI, moveI, adjust_bc, switch_slice, new_point, new_shape, rotate_3D, color_picker } ButtonHandler;

    /**
     * Enumeration to pass to the button handler setting method to know which button a given.
     * is connected to
     */
    typedef enum { left = 0, right = 1 } MouseButton;

    
    /**
     * A list to hold those slots that are implemented in this type of viewer.
     */
    typedef std::list< unsigned > AvailableSlots;

   
    #ifdef _MSC_VER                    // disable a warning type in MSVC++
        #pragma warning(push)
        #pragma warning(disable: 4355) // 'this' : used in base member initializer list
    #endif
    /**
     * Constructor that does nothing except for initializing the _inputPorts.
     */
    m4dGUIAbstractViewerWidget() : _inputPorts( this )
    {
        // connect message receiving signal to message handling slot
	qRegisterMetaType<Imaging::PipelineMsgID>( "Imaging::PipelineMsgID" );
	m4dGUIAbstractViewerWidget::connect( this, SIGNAL(signalMessageHandler( Imaging::PipelineMsgID )), this, SLOT(slotMessageHandler( Imaging::PipelineMsgID )), Qt::QueuedConnection );
	_leftSideData.clear();
	_rightSideData.clear();
    }
    #ifdef _MSC_VER                    // restore the above disabled warning type in MSVC++
        #pragma warning(pop)
    #endif

    /**
     * Virtual destructor that does nothing.
     */
    virtual ~m4dGUIAbstractViewerWidget() {}

    /**
     * Clears the input port - unconnects the viewer.
     */
    virtual void setInputPort()=0;
    
    /**
     * (Re)sets the input port - (re)connects the viewer.
     *  @param conn the connection that connects the viewer
     */
    virtual void setInputPort( Imaging::ConnectionInterface* conn )=0;

    /**
     * Get the connection that connects the viewer.
     *  @return the connection that connects the viewer
     */
    Imaging::ConnectionInterface* getInputPort()
    	{ return _inputPorts[0].GetConnection(); }

    
    /**
     * Return the selected status of the viewer.
     *  @return true if the viewer is selected, false otherwise
     */
    bool getSelected()
	{ return _selected; }
    
    /**
     * Set the viewer to not selected.
     */
    virtual void setUnSelected()=0;

    /**
     * Set the viewer to selected.
     */
    virtual void setSelected()=0;

    /**
     * Find out which viewer slots are implemented in the given viewer.
     *  @return list of integers indicating the implemented viewer slots
     */
    virtual AvailableSlots getAvailableSlots()=0;
    
    /**
     * Cast explicitly the viewer to a QWidget. It is necessary for being able to add
     * the widget to other Qt widgets - this class has only a QObject base; the inheriting
     * class has to inherit from QWidget (the reason for this is the problem of multiple
     * inheritence, since the inheriting class will probably inherit from another subclass
     * of QWidget, like QVTKWidget or QGLWidget).
     *  @return Pointer that is casted to the QWidget base of the implementing class
     */
    virtual QWidget* operator()()=0;

    
    /**
     * Return the list of input ports connected to this viewer.
     *  @return list of connected input ports
     */
    const Imaging::InputPortList &
    InputPort()const
    	{ return _inputPorts; }

    
    /**
     * Return the index of the viewer - it is handy when there are several viewers
     * included in widget.
     *  @return index number
     */
    unsigned getIndex()
    	{ return _index; }
    
    
    /**
     * Set the data text to be printed on the left side of the viewer.
     *  @param dataList a list of the data text
     */
    void setLeftSideTextData( std::list< std::string >& dataList )
    {
        _leftSideData = dataList;
    }
    
    /**
     * Set the data text to be printed on the right side of the viewer.
     *  @param dataList a list of the data text
     */
    void setRightSideTextData( std::list< std::string >& dataList )
    {
        _rightSideData = dataList;
    }

    /**
     * Get the data text to be printed on the left side of the viewer.
     *  @return a list of the data text
     */
    const std::list< std::string >& getLeftSideTextData()const
    {
        return _leftSideData;
    }

    /**
     * Get the data text to be printed on the right side of the viewer.
     *  @return a list of the data text
     */
    const std::list< std::string >& getRightSideTextData()const
    {
        return _rightSideData;
    }

protected:
    
    /**
     * List of the input ports connected to the given viewer.
     */
    Imaging::InputPortList	_inputPorts;

    /**
     * Tells if the viewer is selected or not.
     */
    bool			_selected;

    /**
     * The index of the given viewer.
     */
    unsigned			_index;

    /**
     * The list of text data to be printed on the left side of the viewer.
     */
    std::list< std::string >			_leftSideData;

    /**
     * The list of text data to be printed on the right side of the viewer.
     */
    std::list< std::string >			_rightSideData;

public slots:

    /**
     * Slot to connect a given button to a given handler method.
     *  @param hnd the handler method
     *  @param btn the button to connect to the method
     */
    virtual void slotSetButtonHandler( ButtonHandler hnd, MouseButton btn )=0;
    
    /**
     * Slot to set if the viewer is selected or not.
     *  @param selected tells if the viewer should be selected or not
     */
    virtual void slotSetSelected( bool selected )=0;

    /**
     * Slot to set the current slice number.
     *  @param num the slice number to be set
     */
    virtual void slotSetSliceNum( size_t num )=0;

    /**
     * Slot to set the viewer to show one slice at once.
     */
    virtual void slotSetOneSliceMode()=0;

    /**
     * Slot to set the viewer to show several slices at once.
     *  @param slicesPerRow how many slices will be shown in one row
     *  @param slicesPerColumn how many slices will be shown in one column
     */
    virtual void slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn )=0;

    /**
     * Slot to toggle vertical flip.
     */
    virtual void slotToggleFlipVertical()=0;

    /**
     * Slot to toggle horizontal flip.
     */
    virtual void slotToggleFlipHorizontal()=0;

    /**
     * Slot to add some text data to show on the left side of the viewer.
     *  @param data the value of the given data
     */
    virtual void slotAddLeftSideData( std::string data )=0;

    /**
     * Slot to add some text data to show on the right side of the viewer.
     *  @param data the value of the given data
     */
    virtual void slotAddRightSideData( std::string data )=0;

    /**
     * Slot to clear all data from the left side of the viewer.
     */
    virtual void slotClearLeftSideData()=0;

    /**
     * Slot to clear all data from the right side of the viewer.
     */
    virtual void slotClearRightSideData()=0;

    /**
     * Slot to toggle the printing of data on the viewer.
     */
    virtual void slotTogglePrintData()=0;

    /**
     * Slot to toggle the printing of the selected shapes' information on the viewer.
     */
    virtual void slotTogglePrintShapeData()=0;
    
    /**
     * Slot to zoom the image.
     *  @param amount how much we want to zoom. Positive value means zoom in,
     *                negative value means zoom out.
     */
    virtual void slotZoom( int amount )=0;

    /**
     * Slot to move the image.
     *  @param amountH the amount to move the image horizontally
     *  @param amountV the amount to move the image vertically
     */
    virtual void slotMove( int amountH, int amountV )=0;

    /**
     * Slot to adjust the brightness and contrast of the image.
     *  @param amountB the amount to adjust the brightness
     *  @param amountC the amount to adjust the contrast
     */
    virtual void slotAdjustContrastBrightness( int amountB, int amountC )=0;

    /**
     * Slot to add a new point to the last created shape of the list of selected
     * shapes.
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     *  @param z the z coordinate of the point
     */
    virtual void slotNewPoint( double x, double y, double z )=0;

    /**
     * Slot to add a new shape to the list of selected shapes.
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     *  @param z the z coordinate of the point
     */
    virtual void slotNewShape( double x, double y, double z )=0;

    /**
     * Slot to delete the last selected point.
     */
    virtual void slotDeletePoint()=0;
    
    /**
     * Slot to delete the last selected shape.
     */
    virtual void slotDeleteShape()=0;

    /**
     * Slot to erase all selected shapes and points.
     */
    virtual void slotDeleteAll()=0;

    /**
     * Slot to rotate the scene around the x axis.
     *  @param x the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisX( double x )=0;

    /**
     * Slot to rotate the scene around the y axis.
     *  @param y the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisY( double y )=0;

    /**
     * Slot to rotate the scene around the z axis.
     *  @param z the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisZ( double z )=0;

    /**
     * Slot to toggle the orientation of the slice viewing axes.
     * xy -> yz -> zx
     */
    virtual void slotToggleSliceOrientation()=0;

    /**
     * Slot to pick the color of the pixel at the given position.
     *  @param x the x coordinate
     *  @param y the y coordinate
     *  @param z the z coordinate
     */
    virtual void slotColorPicker( double x, double y, double z )=0;

protected slots:
    
    /**
     * Slot to handle incoming message from Image pipeline.
     *  @param msgID the ID of the message
     */
    virtual void slotMessageHandler( Imaging::PipelineMsgID msgID )=0;

signals:

    /**
     * Signal indicating that a handler has been assigned to a mouse button.
     *  @param index the index of the viewer
     *  @param hnd the method that has been assigned
     *  @param btn the mouse button that has been assigned to
     */
    void signalSetButtonHandler( unsigned index, ButtonHandler hnd, MouseButton btn );

    /**
     * Signal indicating that selected mode has changed.
     *  @param index the index of the viewer
     *  @param selected true if the viewer has been set selected, false otherwise
     */
    void signalSetSelected( unsigned index, bool selected );

    /**
     * Signal indicating that the slice number has been changed.
     *  @param index the index of the viewer
     *  @param num the slice number that has been set
     */
    void signalSetSliceNum( unsigned index, size_t num );

    /**
     * Signal indicating that the viewer is set to show one slice at a time.
     *  @param index the index of the viewer
     */
    void signalSetOneSliceMode( unsigned index );

    /**
     * Signal indicating that the viewer is set to show more slices at a time.
     *  @param index the index of the viewer
     *  @param slicesPerRow the number of slices that should be displayed in a row
     *  @param slicesPerColumn how many slices will be shown in one column
     */
    void signalSetMoreSliceMode( unsigned index, unsigned slicesPerRow, unsigned slicesPerColumn );

    /**
     * Signal indicating that the vertical flip has been toggled.
     */
    void signalToggleFlipVertical();

    /**
     * Signal indicating that the horizontal flip has been toggled.
     */
    void signalToggleFlipHorizontal();

    /**
     * Signal indicating that some text data has been added to the left side of the viewer.
     *  @param data the value of the data
     */
    void signalAddLeftSideData( std::string data );

    /**
     * Signal indicating that some text data has been added to the right side of the viewer.
     *  @param data the value of the data
     */
    void signalAddRightSideData( std::string data );

    /**
     * Signal indicating that all the text data have been erased from the left side of the viewer.
     */
    void signalClearLeftSideData();

    /**
     * Signal indicating that all the text data have been erased from the right side of the viewer.
     */
    void signalClearRightSideData();

    /**
     * Signal indicating that the printing of text data on the sides of the viewer has been toggled.
     */
    void signalTogglePrintData();

    /**
     * Signal indicating that the printing of shape information has been toggled.
     */
    void signalTogglePrintShapeData();

    /**
     * Signal indicating that the viewer's zooming has changed.
     *  @param index the index of the viewer
     *  @param amount the amount by which the zoom rate has change
     */
    void signalZoom( unsigned index, int amount );

    /**
     * Signal indicating that the image's position has changed.
     *  @param index the index of the viewer
     *  @param amountH the horizontal difference between the new and the old position
     *  @param amountV the vertical difference between the new and the old position
     */
    void signalMove( unsigned index, int amountH, int amountV );

    /**
     * Signal indicating that the brightness and/or contrast of the viewer has changed.
     *  @param index the index of the viewer
     *  @param amountB the amount by which the brightness rate has changed
     *  @param amountC the amount by which the contrast rate has changed
     */
    void signalAdjustContrastBrightness( unsigned index, int amountB, int amountC );

    /**
     * Signal indicating that a new point has been added to the selected points of the viewer.
     *  @param index the index of the viewer
     *  @param x the x coordinate of the newly selected point
     *  @param y the y coordinate of the newly selected point
     *  @param z the z coordinate of the newly selected point
     */
    void signalNewPoint( unsigned index, double x, double y, double z );

    /**
     * Signal indicating that a new shape has been added to the selected points of the viewer.
     *  @param index the index of the viewer
     *  @param x the x coordinate of the first point of the new shape
     *  @param y the y coordinate of the first point of the new shape
     *  @param z the z coordinate of the first point of the new shape
     */
    void signalNewShape( unsigned index, double x, double y, double z );

    /**
     * Signal indicating that the last point has been deleted from the selected points of the viewer.
     *  @param index the index of the viewer
     */
    void signalDeletePoint( unsigned index );

    /**
     * Signal indicating that the last shape has been deleted from the selected shapes of the viewer.
     *  @param index the index of the viewer
     */
    void signalDeleteShape( unsigned index );

    /**
     * Signal indicating that all the shapes and points have been deleted from the selected
     * shapes/points of the viewer.
     *  @param index the index of the viewer
     */
    void signalDeleteAll( unsigned index );

    /**
     * Signal indicating that the scene has been rotated around axis x.
     *  @param index the index of the viewer
     *  @param x the angle by which the scene has been rotated
     */
    void signalRotateAxisX( unsigned index, double x );

    /**
     * Signal indicating that the scene has been rotated around axis y.
     *  @param index the index of the viewer
     *  @param y the angle by which the scene has been rotated
     */
    void signalRotateAxisY( unsigned index, double y );

    /**
     * Signal indicating that the scene has been rotated around axis z.
     *  @param index the index of the viewer
     *  @param z the angle by which the scene has been rotated
     */
    void signalRotateAxisZ( unsigned index, double z );

    /**
     * Signal indicating that the slice viewing orientation axis has been toggled.
     * xy -> yz -> zx
     *  @param index the index of the viewer
     */
    void signalToggleSliceOrientation( unsigned index );

    /**
     * Signal indicating that a color has been picked.
     *  @param index the index of the viewer
     *  @param value the value of the color that has been picked
     */
    void signalColorPicker( unsigned index, int64 value );
    
    
    /**
     * Signal indicating that a message has been received.
     *  @param msgID the ID of the message
     */
    void signalMessageHandler( Imaging::PipelineMsgID msgID );

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

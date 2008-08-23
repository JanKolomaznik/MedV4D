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

class m4dGUISliceViewerWidget : public m4dGUIAbstractViewerWidget, public QGLWidget
{
    Q_OBJECT

public:
    
    typedef void (M4D::Viewer::m4dGUISliceViewerWidget::*ButtonMethods)( int amount1, int amount2 );
    typedef void (M4D::Viewer::m4dGUISliceViewerWidget::*SelectMethods)( double x, double y, double z );
    
    m4dGUISliceViewerWidget( unsigned index, QWidget *parent = 0 );
    m4dGUISliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 );
    virtual void setInputPort();
    virtual void setInputPort( Imaging::ConnectionInterface* conn );

    virtual void setUnSelected();
    virtual void setSelected();

    virtual AvailableSlots getAvailableSlots();

    virtual QWidget* operator()();

    virtual void ReceiveMessage( Imaging::PipelineMessage::Ptr msg, Imaging::PipelineMessage::MessageSendStyle sendStyle, Imaging::FlowDirection direction );

public slots:

    /**
     * Slot to connect a given button to a given handler method
     *  @param hnd the handler method
     *  @param btn the button to connect to the method
     */
    virtual void slotSetButtonHandler( ButtonHandler hnd, MouseButton btn );
    
    /**
     * Slot to set if the viewer is selected or not
     *  @param selected tells if the viewer should be selected or not
     */
    virtual void slotSetSelected( bool selected );

    /**
     * Slot to set the current slice number
     *  @param num the slice number to be set
     */
    virtual void slotSetSliceNum( size_t num );

    /**
     * Slot to set the viewer to show one slice at once
     */
    virtual void slotSetOneSliceMode();

    /**
     * Slot to set the viewer to show several slices at once
     *  @param slicesPerRow how many slices will be shown in one row
     *  @param slicesPerColumn how many slices will be shown in one column
     */
    virtual void slotSetMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn );

    /**
     * Slot to toggle vertical flip
     */
    virtual void slotToggleFlipVertical();

    /**
     * Slot to toggle horizontal flip
     */
    virtual void slotToggleFlipHorizontal();

    /**
     * Slot to add some text data to show on the left side of the viewer
     *  @param type the type of the given data
     *  @param data the value of the given data
     */
    virtual void slotAddLeftSideData( std::string type, std::string data );

    /**
     * Slot to add some text data to show on the right side of the viewer
     *  @param type the type of the given data
     *  @param data the value of the given data
     */
    virtual void slotAddRightSideData( std::string type, std::string data );

    /**
     * Slot to erase some data from the left side of the viewer
     *  @param type the type of the data that is to be erased
     */
    virtual void slotEraseLeftSideData( std::string type );

    /**
     * Slot to erase some data from the right side of the viewer
     *  @param type the type of the data that is to be erased
     */
    virtual void slotEraseRightSideData( std::string type );

    /**
     * Slot to clear all data from the left side of the viewer
     */
    virtual void slotClearLeftSideData();

    /**
     * Slot to clear all data from the right side of the viewer
     */
    virtual void slotClearRightSideData();

    /**
     * Slot to toggle the printing of data on the viewer
     */
    virtual void slotTogglePrintData();

    /**
     * Slot to toggle the printing of the selected shapes' information on the viewer
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
     *  @param amountH the amount to move the image horizontally
     *  @param amountV the amount to move the image vertically
     */
    virtual void slotMove( int amountH, int amountV );

    /**
     * Slot to adjust the brightness and contrast of the image
     *  @param amountB the amount to adjust the brightness
     *  @param amountC the amount to adjust the contrast
     */
    virtual void slotAdjustContrastBrightness( int amountB, int amountC );

    /**
     * Slot to add a new point to the last created shape of the list of selected
     * shapes
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     *  @param z the z coordinate of the point
     */
    virtual void slotNewPoint( double x, double y, double z );

    /**
     * Slot to add a new shape to the list of selected shapes
     *  @param x the x coordinate of the point
     *  @param y the y coordinate of the point
     *  @param z the z coordinate of the point
     */
    virtual void slotNewShape( double x, double y, double z );

    /**
     * Slot to delete the last selected point
     */
    virtual void slotDeletePoint();
    
    /**
     * Slot to delete the last selected shape
     */
    virtual void slotDeleteShape();

    /**
     * Slot to erase all selected shapes and poitns
     */
    virtual void slotDeleteAll();

    /**
     * Slot to rotate the scene around the x axis
     * (it has no function in this type of viewer)
     *  @param x the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisX( double x );

    /**
     * Slot to rotate the scene around the y axis
     * (it has no function in this type of viewer)
     *  @param y the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisY( double y );

    /**
     * Slot to rotate the scene around the z axis
     * (it has no function in this type of viewer)
     *  @param z the angle that the scene is to be rotated by
     */
    virtual void slotRotateAxisZ( double z );

    /**
     * Slot to toggle the orientation of the slice viewing axes
     * xy -> yz -> zx
     */
    virtual void slotToggleSliceOrientation();

    /**
     * Slot to pick the color of the pixel at the given position
     *  @param x the x coordinate
     *  @param y the y coordinate
     *  @param z the z coordinate
     */
    virtual void slotColorPicker( double x, double y, double z );

protected slots:
    
    /**
     * Slot to handle incoming message from Image pipeline
     *  @param msgID the ID of the message
     */
    virtual void slotMessageHandler( Imaging::PipelineMsgID msgID );

protected:
    bool checkOutOfBounds( double x, double y );
    void resolveFlips( double& x, double& y );
    void resolveFlips( int& x, int& y );
    void setButtonHandler( ButtonHandler hnd, MouseButton btn );
    void ImagePositionSelectionCaller( int x, int y, SelectMethods f );
    void setOneSliceMode();
    void setMoreSliceMode( unsigned slicesPerRow, unsigned slicesPerColumn );
    void switchSlice( int dummy, int amount );
    void toggleFlipHorizontal();
    void toggleFlipVertical();
    void addLeftSideData( std::string type, std::string data );
    void addRightSideData( std::string type, std::string data );
    void eraseLeftSideData( std::string type );
    void eraseRightSideData( std::string type );
    void clearLeftSideData();
    void clearRightSideData();
    void togglePrintData();
    void paintGL();
    void resizeGL(int winW, int winH);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void setSliceNum( size_t num );
    void zoomImage( int dummy, int amount );
    void moveImage( int amountH, int amountV );
    void adjustContrastBrightness( int amountB, int amountC );
    void newPoint( double x, double y, double z );
    void newShape( double x, double y, double z );
    void deletePoint();
    void deleteShape();
    void deleteAll();
    void colorPicker( double x, double y, double z );

private:

    void setParameters();
    void resetParameters();
    void borderDrawer( GLfloat red, GLfloat green, GLfloat blue, unsigned pos );
    void drawPluggedBorder();
    void drawSelectionModeBorder();
    void drawSelectedBorder();
    void drawData( double zoomRate, QPoint offset, int sliceNum );
    void drawSlice( int sliceNum, double zoomRate, QPoint offset );
    void drawShape( Selection::m4dShape<double>& s, bool last, int sliceNum, float zoomRate );
    void drawPicked();
    void calculateWidthHeight( double& w, double& h );
    void calculateOptimalZoomRate();

    Imaging::InputPortAbstractImage*		_inPort;

    std::list< Selection::m4dShape<double> >	_shapes;

    std::map< std::string, std::string >	_leftSideData;
    std::map< std::string, std::string >	_rightSideData;

    short					_flipH;
    short					_flipV;
    size_t					_minimum[3];
    size_t					_maximum[3];
    float					_extents[3];
    int						_imageID;

    bool					_ready;
    bool					_printData;
    bool					_printShapeData;
    bool					_oneSliceMode;
    QPoint					_lastPos;
    QPoint					_offset;
    int						_sliceNum;
    unsigned					_slicesPerRow;
    unsigned					_slicesPerColumn;
    double					_zoomRate;
    GLint					_brightnessRate;
    GLfloat					_contrastRate;
    ButtonMethods				_buttonMethods[2];
    SelectMethods				_selectMethods[2];
    bool					_selectionMode[2];
    AvailableSlots				_availableSlots;
    SliceOrientation				_sliceOrientation;
    bool					_colorPicker;
    int64					_colorPicked;
    QPoint					_pickedPosition;
    int						_slicePicked;
};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

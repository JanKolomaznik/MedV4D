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
    virtual void slotSetButtonHandler( ButtonHandler hnd, MouseButton btn );
    virtual void slotSetSelected( bool selected );
    virtual void slotSetSliceNum( size_t num );
    virtual void slotSetOneSliceMode();
    virtual void slotSetMoreSliceMode( unsigned slicesPerRow );
    virtual void slotToggleFlipVertical();
    virtual void slotToggleFlipHorizontal();
    virtual void slotAddLeftSideData( std::string type, std::string data );
    virtual void slotAddRightSideData( std::string type, std::string data );
    virtual void slotEraseLeftSideData( std::string type );
    virtual void slotEraseRightSideData( std::string type );
    virtual void slotClearLeftSideData();
    virtual void slotClearRightSideData();
    virtual void slotTogglePrintData();
    virtual void slotTogglePrintShapeData();
    virtual void slotZoom( int amount );
    virtual void slotMove( int amountH, int amountV );
    virtual void slotAdjustContrastBrightness( int amountB, int amountC );
    virtual void slotNewPoint( double x, double y, double z );
    virtual void slotNewShape( double x, double y, double z );
    virtual void slotDeletePoint();
    virtual void slotDeleteShape();
    virtual void slotDeleteAll();
    virtual void slotRotateAxisX( double x );
    virtual void slotRotateAxisY( double y );
    virtual void slotRotateAxisZ( double z );
    virtual void slotToggleSliceOrientation();
    virtual void slotColorPicker( double x, double y, double z );

protected slots:
    void slotMessageHandler( Imaging::PipelineMsgID msgID );

protected:
    bool checkOutOfBounds( double x, double y );
    void resolveFlips( double& x, double& y );
    void resolveFlips( int& x, int& y );
    void setButtonHandler( ButtonHandler hnd, MouseButton btn );
    void ImagePositionSelectionCaller( int x, int y, SelectMethods f );
    void setOneSliceMode();
    void setMoreSliceMode( unsigned slicesPerRow );
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
    void drawData( double zoomRate, QPoint offset );
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
    double					_zoomRate;
    GLint					_brightnessRate;
    GLfloat					_contrastRate;
    ButtonMethods				_buttonMethods[2];
    SelectMethods				_selectMethods[2];
    bool					_selectionMode[2];
    AvailableSlots				_availableSlots;
    SliceOrientation				_sliceOrientation;
    bool					_colorPicker;
    uint64					_colorPicked;
    QPoint					_pickedPosition;
    int						_slicePicked;
};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

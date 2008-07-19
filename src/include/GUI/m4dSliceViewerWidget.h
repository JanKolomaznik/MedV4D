#ifndef _M4DSLICEVIEWERWIDGET_H
#define _M4DSLICEVIEWERWIDGET_H

#include <QtOpenGL>
#include <list>
#include <string>
#include <map>
#include "Imaging/Image.h"
#include "Common.h"
#include "ExceptionBase.h"
#include "Imaging/Ports.h"
#include "Imaging/DefaultConnection.h"
#include "GUI/m4dSelection.h"
#include "GUI/m4dAbstractViewerWidget.h"

#define RW 0.3086
#define GW 0.6094
#define BW 0.0820

namespace M4D
{
namespace Viewer
{

class m4dSliceViewerWidget : public m4dAbstractViewerWidget, public QGLWidget
{
    Q_OBJECT

public:
    m4dSliceViewerWidget( unsigned index, QWidget *parent = 0 );
    m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn, unsigned index, QWidget *parent = 0 );
    m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn, unsigned index, QWidget *parent = 0 );
    m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn, unsigned index, QWidget *parent = 0 );
    ~m4dSliceViewerWidget();
    Imaging::InputPortAbstractImage& getInputPort();
    void setInputPort();
    void setInputPort( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn );
    void setInputPort( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn );
    void setInputPort( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn );

    virtual AvailableSlots getAvailableSlots();

    virtual QWidget* operator()();

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
    virtual void slotZoom( int amount );
    virtual void slotMove( int amountH, int amountV );
    virtual void slotAdjustContrastBrightness( int amountB, int amountC );
    virtual void slotNewPoint( int x, int y, int z );
    virtual void slotNewShape( int x, int y, int z );
    virtual void slotDeletePoint();
    virtual void slotDeleteShape();
    virtual void slotRotateAxisX( int x );
    virtual void slotRotateAxisY( int y );
    virtual void slotRotateAxisZ( int z );

protected:
    void setButtonHandler( ButtonHandler hnd, MouseButton btn );
    void setUnSelected();
    void setSelected();
    bool getSelected();
    void setOneSliceMode();
    void setMoreSliceMode( unsigned slicesPerRow );
    void setColorMode( ColorMode cm );
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
    void newPoint( int x, int y, int z );
    void newShape( int x, int y, int z );
    void deletePoint( int x, int y, int z );
    void deleteShape( int x, int y, int z );

private:

    void setParameters();
    void drawSelectionModeBorder();
    void drawSelectedBorder();
    void drawData( double zoomRate, QPoint offset );
    void drawSlice( int sliceNum, double zoomRate, QPoint offset );
    void drawShape( Selection::m4dShape<int>& s, bool last, int sliceNum, float zoomRate );

    typedef void (M4D::Viewer::m4dSliceViewerWidget::*ButtonMethods)( int amount1, int amount2 );
    typedef void (M4D::Viewer::m4dSliceViewerWidget::*SelectMethods)( int x, int y, int z );
    
    Imaging::InputPortAbstractImage		_inPort;

    std::list< Selection::m4dShape<int> >	_shapes;

    std::map< std::string, std::string >	_leftSideData;
    std::map< std::string, std::string >	_rightSideData;

    unsigned					_index;

    short					_flipH;
    short					_flipV;
    
    bool					_printData;
    bool					_printShapeData;
    bool					_oneSliceMode;
    bool					_selected;
    ColorMode					_colorMode;
    QPoint					_lastPos;
    QPoint					_offset;
    int						_sliceNum;
    unsigned					_slicesPerRow;
    double					_zoomRate;
    GLfloat					_brightnessRate;
    GLfloat					_contrastRate;
    ButtonMethods				_buttonMethods[2];
    SelectMethods				_selectMethods[2];
    bool					_selectionMode[2];
    AvailableSlots				_availableSlots;
};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

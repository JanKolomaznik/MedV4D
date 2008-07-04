#ifndef _M4DSLICEVIEWERWIDGET_H
#define _M4DSLICEVIEWERWIDGET_H

#include <QGLWidget>
#include "Imaging/Image.h"
#include "Common.h"
#include "ExceptionBase.h"
#include "Imaging/Ports.h"
#include "Imaging/DefaultConnection.h"
#include "GUI/m4dSelection.h"

#define RW 0.3086
#define GW 0.6094
#define BW 0.0820

namespace M4D
{
namespace Viewer
{

class m4dSliceViewerWidget : public QGLWidget
{
    Q_OBJECT

public:
    m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn, QWidget *parent = 0 );
    m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn, QWidget *parent = 0 );
    m4dSliceViewerWidget( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn, QWidget *parent = 0 );
    ~m4dSliceViewerWidget();
    Imaging::InputPortAbstractImage& getInputPort();
    void setInputPort( Imaging::ImageConnection< Imaging::Image< uint32, 3 > >& conn );
    void setInputPort( Imaging::ImageConnection< Imaging::Image< uint16, 3 > >& conn );
    void setInputPort( Imaging::ImageConnection< Imaging::Image< uint8, 3 > >& conn );

    typedef enum { NONE_BUTTON, ZOOM, MOVE_H, MOVE_V, ADJUST_B, ADJUST_C } ButtonHandlers;
    typedef enum { NONE_SELECT, NEW_POINT, NEW_SHAPE, DELETE_POINT, DELETE_SHAPE } SelectHandlers;
    typedef enum { RGBA_UNSIGNED_BYTE, 	GRAYSCALE_UNSIGNED_BYTE, GRAYSCALE_UNSIGNED_SHORT } ColorMode;

    void setButtonHandlers( ButtonHandlers* hnd );
    void setSelectHandlers( SelectHandlers* hnd );
    void setSelectionMode( bool mode );
    bool getSelectionMode();

protected:
    void setColorMode( ColorMode cm );
    void paintGL();
    void resizeGL(int winW, int winH);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void setSliceNum( size_t num );
    void zoomImage( int amount );
    void moveImageH( int amount );
    void moveImageV( int amount );
    void adjustBrightness( int amount );
    void adjustContrast( int amount );
    void none( int amount );
    void nonePos( int x, int y, int z );
    void newPoint( int x, int y, int z );
    void newShape( int x, int y, int z );
    void deletePoint( int x, int y, int z );
    void deleteShape( int x, int y, int z );

private:

    void setParameters();
    void drawBorder();
    void drawShape( Selection::m4dShape<int>& s, bool last );

    typedef void (M4D::Viewer::m4dSliceViewerWidget::*ButtonMethods)( int amount );
    typedef void (M4D::Viewer::m4dSliceViewerWidget::*SelectMethods)( int x, int y, int z );
    
    Imaging::InputPortAbstractImage		_inPort;

    std::list<Selection::m4dShape<int> >	_shapes;

    bool					_selectionMode;
    bool					_printShapeData;
    ColorMode					_colorMode;
    QPoint					_lastPos;
    QPoint					_offset;
    int						_sliceNum;
    double					_zoomRate;
    GLfloat					_brightnessRate;
    GLfloat					_contrastRate;
    ButtonMethods				_buttonMethods[3][2];
    SelectMethods				_selectMethods[2][2];
};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

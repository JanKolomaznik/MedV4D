#ifndef _M4DABSTRACTVIEWERWIDGET_H
#define _M4DABSTRACTVIEWERWIDGET_H

#include "Common.h"
#include <QWidget>

#define  SETBUTTONHANDLERS	1
#define  SETSELECTHANDLERS	2
#define  SETSELECTIONMODE	4
#define  SETCOLORMODE		8
#define  SETSLICENUM		16
#define  ZOOM			32
#define  MOVEH			64
#define  MOVEV			128
#define  ADJUSTBRIGHTNESS	256
#define  ADJUSTCONTRAST		512
#define  NEWPOINT		1024
#define  NEWSHAPE		2048
#define  DELETEPOINT		4096
#define  DELETESHAPE		8192
#define  ROTATEAXISX		16384
#define  ROTATEAXISY		32768
#define  ROTATEAXISZ		65536
#define  SETSELECTED		131072
#define  SETONESLICEMODE	262144
#define  SETMORESLICEMODE	524288

namespace M4D
{
namespace Viewer
{

class m4dAbstractViewerWidget : public QObject
{
    Q_OBJECT

public:
    m4dAbstractViewerWidget() { }
    virtual ~m4dAbstractViewerWidget() { }
    
    typedef enum { none_button, zoom, move_h, move_v, adjust_b, adjust_c } ButtonHandlers;
    typedef enum { none_select, new_point, new_shape, delete_point, delete_shape } SelectHandlers;
    typedef enum { rgba_unsigned_byte, grayscale_unsigned_byte, grayscale_unsigned_short } ColorMode;

    typedef long int AvailableSlots;

    virtual AvailableSlots getAvailableSlots()=0;
    virtual QWidget* operator()()=0;

public slots:
    virtual void slotSetButtonHandlers( ButtonHandlers* hnd )=0;
    virtual void slotSetSelectHandlers( SelectHandlers* hnd )=0;
    virtual void slotSetSelectionMode( bool mode )=0;
    virtual void slotSetSelected( bool selected )=0;
    virtual void slotSetColorMode( ColorMode cm )=0;
    virtual void slotSetSliceNum( size_t num )=0;
    virtual void slotSetOneSliceMode()=0;
    virtual void slotSetMoreSliceMode( unsigned slicesPerRow )=0;
    virtual void slotZoom( int amount )=0;
    virtual void slotMoveH( int amount )=0;
    virtual void slotMoveV( int amount )=0;
    virtual void slotAdjustBrightness( int amount )=0;
    virtual void slotAdjustContrast( int amount )=0;
    virtual void slotNewPoint( int x, int y, int z )=0;
    virtual void slotNewShape( int x, int y, int z )=0;
    virtual void slotDeletePoint()=0;
    virtual void slotDeleteShape()=0;
    virtual void slotRotateAxisX( int x )=0;
    virtual void slotRotateAxisY( int y )=0;
    virtual void slotRotateAxisZ( int z )=0;

signals:
    virtual void signalSetButtonHandlers( ButtonHandlers* hnd );
    virtual void signalSetSelectHandlers( SelectHandlers* hnd );
    virtual void signalSetSelectionMode( bool mode );
    virtual void signalSetSelected( bool selected );
    virtual void signalSetColorMode( ColorMode cm );
    virtual void signalSetSliceNum( size_t num );
    virtual void signalSetOneSliceMode();
    virtual void signalSetMoreSliceMode( unsigned slicesPerRow );
    virtual void signalZoom( int amount );
    virtual void signalMoveH( int amount );
    virtual void signalMoveV( int amount );
    virtual void signalAdjustBrightness( int amount );
    virtual void signalAdjustContrast( int amount );
    virtual void signalNewPoint( int x, int y, int z );
    virtual void signalNewShape( int x, int y, int z );
    virtual void signalDeletePoint();
    virtual void signalDeleteShape();
    virtual void signalRotateAxisX( int x );
    virtual void signalRotateAxisY( int y );
    virtual void signalRotateAxisZ( int z );

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

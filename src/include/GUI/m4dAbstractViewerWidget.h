#ifndef _M4DABSTRACTVIEWERWIDGET_H
#define _M4DABSTRACTVIEWERWIDGET_H

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

namespace M4D
{
namespace Viewer
{

class m4dAbstractViewerWidget
{
    Q_OBJECT

public:
    typedef enum { NONE_BUTTON, ZOOM, MOVE_H, MOVE_V, ADJUST_B, ADJUST_C } ButtonHandlers;
    typedef enum { NONE_SELECT, NEW_POINT, NEW_SHAPE, DELETE_POINT, DELETE_SHAPE } SelectHandlers;
    typedef enum { RGBA_UNSIGNED_BYTE, 	GRAYSCALE_UNSIGNED_BYTE, GRAYSCALE_UNSIGNED_SHORT } ColorMode;

    typedef AvailableSlots uint64;

    virtual AvailableSlots getAvailableSlots()=0;

public slots:
    virtual void slotSetButtonHandlers( ButtonHandlers* hnd )=0;
    virtual void slotSetSelectHandlers( SelectHandlers* hnd )=0;
    virtual void slotSetSelectionMode( bool mode )=0;
    virtual void slotSetColorMode( ColorMode cm )=0;
    virtual void slotSetSliceNum( size_t num )=0;
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
    void slotSetButtonHandlers( ButtonHandlers* hnd );
    void slotSetSelectHandlers( SelectHandlers* hnd );
    void slotSetSelectionMode( bool mode );
    void slotSetColorMode( ColorMode cm );
    void slotSetSliceNum( size_t num );
    void slotZoom( int amount );
    void slotMoveH( int amount );
    void slotMoveV( int amount );
    void slotAdjustBrightness( int amount );
    void slotAdjustContrast( int amount );
    void slotNewPoint( int x, int y, int z );
    void slotNewShape( int x, int y, int z );
    void slotDeletePoint( int x, int y, int z );
    void slotDeleteShape( int x, int y, int z );
    void slotRotateAxisX( int x );
    void slotRotateAxisY( int y );
    void slotRotateAxisZ( int z );

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

#ifndef _M4DGUIABSTRACTVIEWERWIDGET_H
#define _M4DGUIABSTRACTVIEWERWIDGET_H

#include "Common.h"
#include "Imaging/PipelineMessages.h"
#include "Imaging/Ports.h"
#include <QWidget>


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
#define ZOOM			14
#define MOVE			15
#define CONTRASTBRIGHTNESS	16
#define NEWPOINT		17
#define NEWSHAPE		18
#define DELETEPOINT		19
#define DELETESHAPE		20
#define DELETEALL		21
#define ROTATEAXISX		22
#define ROTATEAXISY		23
#define ROTATEAXISZ		24

namespace M4D
{
namespace Viewer
{

class m4dGUIAbstractViewerWidget : public QObject, public Imaging::MessageReceiverInterface
{
    Q_OBJECT

public:
    typedef enum { zoomI, moveI, adjust_bc, switch_slice, new_point, new_shape } ButtonHandler;
    typedef enum { left = 0, right = 1 } MouseButton;

    typedef std::list< unsigned > AvailableSlots;

    m4dGUIAbstractViewerWidget() : _inputPorts( this ) {}
    virtual ~m4dGUIAbstractViewerWidget() {}

    virtual void setInputPort()=0;
    virtual void setInputPort( Imaging::ConnectionInterface* conn )=0;
    Imaging::ConnectionInterface* getInputPort()
    	{ return _inputPorts[0].GetConnection(); }

    bool getSelected()
	{ return _selected; }
    virtual void setUnSelected()=0;
    virtual void setSelected()=0;

    virtual AvailableSlots getAvailableSlots()=0;
    virtual QWidget* operator()()=0;

    const Imaging::InputPortList &
    InputPort()const
    	{ return _inputPorts; }

    unsigned getIndex()
    	{ return _index; }

protected:
    Imaging::InputPortList	_inputPorts;
    bool			_selected;
    unsigned			_index;

public slots:
    virtual void slotSetButtonHandler( ButtonHandler hnd, MouseButton btn )=0;
    virtual void slotSetSelected( bool selected )=0;
    virtual void slotSetSliceNum( size_t num )=0;
    virtual void slotSetOneSliceMode()=0;
    virtual void slotSetMoreSliceMode( unsigned slicesPerRow )=0;
    virtual void slotToggleFlipVertical()=0;
    virtual void slotToggleFlipHorizontal()=0;
    virtual void slotAddLeftSideData( std::string type, std::string data )=0;
    virtual void slotAddRightSideData( std::string type, std::string data )=0;
    virtual void slotEraseLeftSideData( std::string type )=0;
    virtual void slotEraseRightSideData( std::string type )=0;
    virtual void slotClearLeftSideData()=0;
    virtual void slotClearRightSideData()=0;
    virtual void slotTogglePrintData()=0;
    virtual void slotZoom( int amount )=0;
    virtual void slotMove( int amountH, int amountV )=0;
    virtual void slotAdjustContrastBrightness( int amountB, int amountC )=0;
    virtual void slotNewPoint( int x, int y, int z )=0;
    virtual void slotNewShape( int x, int y, int z )=0;
    virtual void slotDeletePoint()=0;
    virtual void slotDeleteShape()=0;
    virtual void slotDeleteAll()=0;
    virtual void slotRotateAxisX( int x )=0;
    virtual void slotRotateAxisY( int y )=0;
    virtual void slotRotateAxisZ( int z )=0;

signals:
    void signalSetButtonHandler( unsigned index, ButtonHandler hnd, MouseButton btn );
    void signalSetSelected( unsigned index, bool selected );
    void signalSetSliceNum( unsigned index, size_t num );
    void signalSetOneSliceMode( unsigned index );
    void signalSetMoreSliceMode( unsigned index, unsigned slicesPerRow );
    void signalToggleFlipVertical();
    void signalToggleFlipHorizontal();
    void signalAddLeftSideData( std::string type, std::string data );
    void signalAddRightSideData( std::string type, std::string data );
    void signalEraseLeftSideData( std::string type );
    void signalEraseRightSideData( std::string type );
    void signalClearLeftSideData();
    void signalClearRightSideData();
    void signalTogglePrintData();
    void signalZoom( unsigned index, int amount );
    void signalMove( unsigned index, int amountH, int amountV );
    void signalAdjustContrastBrightness( unsigned index, int amountB, int amountC );
    void signalNewPoint( unsigned index, int x, int y, int z );
    void signalNewShape( unsigned index, int x, int y, int z );
    void signalDeletePoint( unsigned index );
    void signalDeleteShape( unsigned index );
    void signalDeleteAll( unsigned index );
    void signalRotateAxisX( unsigned index, int x );
    void signalRotateAxisY( unsigned index, int y );
    void signalRotateAxisZ( unsigned index, int z );

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

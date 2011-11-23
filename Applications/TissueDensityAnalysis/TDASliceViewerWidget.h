/**
 * @ingroup TDA 
 * @author Milan Lepik
 * @file TDASliceViewerWidget.h 
 * @{ 
 **/

#ifndef _TDASLICEVIEWERWIDGET_H
#define _TDASLICEVIEWERWIDGET_H

#include "m4dGUISliceViewerWidget.h"
#include "TDASliceViewerTexturePreparer.h"
#include "TDASliceViewerSpecialStateOperator.h"

using namespace M4D;
using namespace M4D::Viewer;

class TDASliceViewerWidget;

typedef TDASliceViewerWidget SliceViewer;
typedef boost::shared_ptr< TDASliceViewerSpecialStateOperator >	TDASliceViewerSpecialStateOperatorPtr;

class TDASliceViewerWidget: public m4dGUISliceViewerWidget, QObject
{
    Q_OBJECT

public:
	typedef m4dGUISliceViewerWidget	PredecessorType;

    /**
     * Constructor.
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    TDASliceViewerWidget( unsigned index, QWidget *parent = 0 );

    /**
     * Construtor.
     *  @param conn the connection that connects the viewer
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    TDASliceViewerWidget( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 );

	
	void setMaskConnection(Imaging::ConnectionInterface* connMask);

	//void drawSliceAdditionals( int sliceNum, double zoomRate );

	void drawHUD( int sliceNum, double zoomRate, QPoint offset );

	void setSpecialState( TDASliceViewerSpecialStateOperatorPtr state )
		{ _specialState = state; }

	void setButtonHandler( ButtonHandler hnd, MouseButton btn );
	
	void makeConnections();

public slots:

	void slotSetSpecialStateSelectMethodLeft();
	
	/**
     * Starts selecting a new shape in the image.
     *  @param x the x coordinate of the first point of the shape
     *  @param y the y coordinate of the first point of the shape
     *  @param z the z coordinate of the first point of the shape
     */
    void sphereCenter( double x, double y, double z );


signals:
	
	void signalSphereCenter(double x, double y, double z );
	
protected: 
	/*
	*	do tohoto portu se zapoji maska zobrazeni
	*/
	Imaging::InputPortTyped<Imaging::AImage>* _inMaskPort;

	void specialStateButtonMethodLeft( int amountA, int amountB );

	void specialStateSelectMethodLeft( double x, double y, double z );

	TDASliceViewerSpecialStateOperatorPtr _specialState;

    /**
     * Draws a slice.
     *  @param sliceNum the number of the slice that is to be drawn
     *  @param zoomRate the zoom rate that is to be applied to the image
     *  @param offset the offset of the image on the viewer
     */
    void drawSlice( int sliceNum, double zoomRate, QPoint offset );

};


#endif

/** @} */


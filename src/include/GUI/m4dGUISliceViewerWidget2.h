/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUISliceViewerWidget2.h 
 * @{ 
 **/

#ifndef _M4DGUISLICEVIEWERWIDGET2_H
#define _M4DGUISLICEVIEWERWIDGET2_H

#include "GUI/m4dGUISliceViewerWidget.h"

#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace Viewer
{

class m4dGUISliceViewerWidget2;

typedef m4dGUISliceViewerWidget2 SliceViewer;

class SliceViewerSpecialStateOperator
{
public:
	virtual void
	Draw( SliceViewer & viewer, int sliceNum, double zoomRate ) = 0;

	virtual void 
	ButtonMethodRight( int amountH, int amountV, double zoomRate ) = 0;
	
	virtual void 
	ButtonMethodLeft( int amountH, int amountV, double zoomRate ) = 0;
	
	virtual void 
	SelectMethodRight( double x, double y, int sliceNum ) = 0;
	
	virtual void 
	SelectMethodLeft( double x, double y, int sliceNum ) = 0;
};

typedef boost::shared_ptr< SliceViewerSpecialStateOperator >	SliceViewerSpecialStateOperatorPtr;


class m4dGUISliceViewerWidget2 : public m4dGUISliceViewerWidget
{
    Q_OBJECT

public:
	typedef m4dGUISliceViewerWidget	PredecessorType;

    /**
     * Constructor.
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    m4dGUISliceViewerWidget2( unsigned index, QWidget *parent = 0 ): PredecessorType( index, parent )
	{}

    /**
     * Construtor.
     *  @param conn the connection that connects the viewer
     *  @param index the index of the viewer
     *  @param parent the parent widget of the viewer
     */
    m4dGUISliceViewerWidget2( Imaging::ConnectionInterface* conn, unsigned index, QWidget *parent = 0 ): PredecessorType( conn, index, parent )
	{}

	void drawSliceAdditionals( int sliceNum, double zoomRate );

	void drawHUD( int sliceNum, double zoomRate, QPoint offset );

	void setSpecialState( SliceViewerSpecialStateOperatorPtr state )
		{ _specialState = state; }

protected:    
	void specialStateButtonMethodLeft( int amountA, int amountB );

	void specialStateButtonMethodRight( int amountA, int amountB );

	void specialStateSelectMethodLeft( double x, double y, double z );

	void specialStateSelectMethodRight( double x, double y, double z );

	void setButtonHandler( ButtonHandler hnd, MouseButton btn );
	
	SliceViewerSpecialStateOperatorPtr		_specialState;
};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif

/** @} */


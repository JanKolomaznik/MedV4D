/**
 *  @ingroup gui
 *  @file m4dGUISliceViewerWidget2.cpp
 *  @brief some brief
 */
#include "GUI/m4dGUISliceViewerWidget2.h"


namespace M4D
{
namespace Viewer
{

void 
m4dGUISliceViewerWidget2::specialStateButtonMethodLeft( int amountA, int amountB )
{
	if( _specialState ) {
		_specialState->ButtonMethodLeft( amountA, amountB );
	}
}

void 
m4dGUISliceViewerWidget2::specialStateButtonMethodRight( int amountA, int amountB )
{
	if( _specialState ) {
		_specialState->ButtonMethodRight( amountA, amountB );
	}
}

void 
m4dGUISliceViewerWidget2::specialStateSelectMethodLeft( double x, double y, double z )
{
	if( _specialState ) {
		_specialState->SelectMethodLeft( x, y, z );
	}
}

void 
m4dGUISliceViewerWidget2::specialStateSelectMethodRight( double x, double y, double z )
{
	if( _specialState ) {
		_specialState->SelectMethodRight( x, y, z );
	}
}

void 
m4dGUISliceViewerWidget2::drawSliceAdditionals( int sliceNum, double zoomRate )
{
	if( _specialState ) {
		_specialState->Draw( *this, sliceNum, zoomRate );
	}
	PredecessorType::drawSliceAdditionals( sliceNum, zoomRate );
}

void 
m4dGUISliceViewerWidget2::drawHUD( int sliceNum, double zoomRate, QPoint offset )
{
	PredecessorType::drawHUD( sliceNum, zoomRate, offset );
}

void 
m4dGUISliceViewerWidget2::setButtonHandler( ButtonHandler hnd, MouseButton btn )
{
	if( hnd != specialState ) {
		PredecessorType::setButtonHandler( hnd, btn );
		return;
	}

	_selectMethods[ left ] = (SelectMethods)&M4D::Viewer::m4dGUISliceViewerWidget2::specialStateSelectMethodLeft;
	_selectionMode[ left ] = true;
	_selectMethods[ right ] = (SelectMethods)&M4D::Viewer::m4dGUISliceViewerWidget2::specialStateSelectMethodRight;
	_selectionMode[ right ] = true;

	_buttonMethods[ left ] = (ButtonMethods)&M4D::Viewer::m4dGUISliceViewerWidget2::specialStateButtonMethodLeft;
	_buttonMode[ left ] = true;
	_buttonMethods[ right ] = (ButtonMethods)&M4D::Viewer::m4dGUISliceViewerWidget2::specialStateButtonMethodRight;
	_buttonMode[ right ] = true;

	if ( _ready ) updateGL();
	emit signalSetButtonHandler( _index, hnd, btn );
}

} /*namespace Viewer*/
} /*namespace M4D*/

/**
 *  @ingroup gui
 *  @file m4dGUISliceViewerWidget2.cpp
 *  @brief some brief
 */
#include "m4dMySliceViewerWidget.h"


namespace M4D
{
namespace Viewer
{

void 
m4dMySliceViewerWidget::specialStateButtonMethodLeft( int amountA, int amountB )
{
	if( _specialState ) {
		_specialState->ButtonMethodLeft( amountA, amountB, _zoomRate );
	}
}

void 
m4dMySliceViewerWidget::specialStateButtonMethodRight( int amountA, int amountB )
{
	if( _specialState ) {
		_specialState->ButtonMethodRight( amountA, amountB, _zoomRate );
	}
}

void 
m4dMySliceViewerWidget::specialStateSelectMethodLeft( double x, double y, double z )
{
	if( _specialState ) {
		resolveFlips( x, y );
		int sliceNum = z/_extents[2];
		_specialState->SelectMethodLeft( x + _extents[0]*_minimum[0], y + _extents[1]*_minimum[1], sliceNum, _zoomRate );
	}
}

void 
m4dMySliceViewerWidget::specialStateSelectMethodRight( double x, double y, double z )
{
	if( _specialState ) {
		resolveFlips( x, y );
		int sliceNum = z/_extents[2];
		_specialState->SelectMethodRight( x + _extents[0]*_minimum[0], y + _extents[1]*_minimum[1], sliceNum, _zoomRate );
	}
}

void 
m4dMySliceViewerWidget::drawSliceAdditionals( int sliceNum, double zoomRate )
{
	if( _specialState ) {
		glPushMatrix();

		glTranslatef( -_extents[0]*_minimum[0], -_extents[1]*_minimum[1], 0.0f );

		_specialState->Draw( *this, sliceNum, zoomRate );
		
		glPopMatrix();
	}
	PredecessorType::drawSliceAdditionals( sliceNum, zoomRate );
}

void 
m4dMySliceViewerWidget::drawHUD( int sliceNum, double zoomRate, QPoint offset )
{
	PredecessorType::drawHUD( sliceNum, zoomRate, offset );
}

void 
m4dMySliceViewerWidget::setButtonHandler( ButtonHandler hnd, MouseButton btn )
{
	if( hnd != specialState ) {
		PredecessorType::setButtonHandler( hnd, btn );
		return;
	}

	_selectMethods[ left ] = (SelectMethods)&M4D::Viewer::m4dMySliceViewerWidget::specialStateSelectMethodLeft;
	_selectMode[ left ] = true;
	_selectMethods[ right ] = (SelectMethods)&M4D::Viewer::m4dMySliceViewerWidget::specialStateSelectMethodRight;
	_selectMode[ right ] = true;

	_buttonMethods[ left ] = (ButtonMethods)&M4D::Viewer::m4dMySliceViewerWidget::specialStateButtonMethodLeft;
	_buttonMode[ left ] = true;
	_buttonMethods[ right ] = (ButtonMethods)&M4D::Viewer::m4dMySliceViewerWidget::specialStateButtonMethodRight;
	_buttonMode[ right ] = true;

	if ( _ready ) updateGL();
	emit signalSetButtonHandler( _index, hnd, btn );
}

void 
m4dMySliceViewerWidget::mousePressEvent(QMouseEvent *event) {
  MessageBox(NULL, L"AHOJ", L"AHOJ",MB_OK);
}


} /*namespace Viewer*/
} /*namespace M4D*/

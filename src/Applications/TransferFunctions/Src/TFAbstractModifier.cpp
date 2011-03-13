#include "TFAbstractModifier.h"

namespace M4D {
namespace GUI {

TFAbstractModifier::TFAbstractModifier():
	toolsWidget_(NULL){
}

TFAbstractModifier::~TFAbstractModifier(){}
/*
void TFAbstractModifier::setWorkCopy(TFWorkCopy::Ptr workCopy){

	workCopy_ = workCopy;
}
*/
QWidget* TFAbstractModifier::getTools(){

	return toolsWidget_;
}

TFWorkCopy::Ptr TFAbstractModifier::getWorkCopy(){

	return workCopy_;
}

void TFAbstractModifier::setInputArea(QRect inputArea){

	inputArea_ = inputArea;
	workCopy_->resize(inputArea_.width(), inputArea_.height());
}

M4D::Common::TimeStamp TFAbstractModifier::getLastChangeTime(){

	return lastChange_;
}

TFPaintingPoint TFAbstractModifier::getRelativePoint_(const TFSize& x, const TFSize& y){	

	int xMax = inputArea_.width() - 1;
	int yMax = inputArea_.height();
	
	TFPaintingPoint corrected = TFPaintingPoint(x - inputArea_.x(), inputArea_.height() - (y - inputArea_.y()));

	if( corrected.x < 0 )
	{
		corrected.x = 0;
	}
	if( corrected.x > xMax )
	{
		corrected.x = xMax;
	}
	if( corrected.y < 0 )
	{
		corrected.y = 0;
	}
	if( corrected.y > yMax )
	{
		corrected.y = yMax;
	}
		
	return corrected;
}

void TFAbstractModifier::addLine_(TFPaintingPoint begin, TFPaintingPoint end){
	
	addLine_(begin.x, begin.y, end.x, end.y);
}

void TFAbstractModifier::addLine_(int x1, int y1, int x2, int y2){ // assumes x1<x2, |y2-y1|<|x2-x1|
	
	//tfAssert((x1 < x2) && (abs(y2-y1) < abs(x2-x1)));
	if(x1==x2 && y1==y2) addPoint_(x1,y1);

    int D, ax, ay, sx, sy;

    sx = x2 - x1;
    ax = abs( sx ) << 1;

    if ( sx < 0 ) sx = -1;
    else if ( sx > 0 ) sx = 1;

    sy = y2 - y1;
    ay = abs( sy ) << 1;

    if ( sy < 0 ) sy = -1;
    else if ( sy > 0 ) sy = 1;

    if ( ax > ay )                          // x coordinate is dominant
    {
		D = ay - (ax >> 1);                   // initial D
		ax = ay - ax;                         // ay = increment0; ax = increment1

		while ( x1 != x2 )
		{
			addPoint_(x1,y1);
			if ( D >= 0 )                       // lift up the Y coordinate
			{
				y1 += sy;
				D += ax;
			}
			else
			{
				D += ay;
			}
			x1 += sx;
		}
    }
    else                                    // y coordinate is dominant
    {
		D = ax - (ay >> 1);                   // initial D
		ay = ax - ay;                         // ax = increment0; ay = increment1

		while ( y1 != y2 )
		{
			addPoint_(x1,y1);
			if ( D >= 0 )                       // lift up the X coordinate
			{
				x1 += sx;
				D += ay;
			}
			else
			{
				D += ax;
			}
			y1 += sy;
		}
    }
}

} // namespace GUI
} // namespace M4D

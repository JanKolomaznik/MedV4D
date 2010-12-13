#include "TFGrayscaleAlphaPainter.h"

namespace M4D {
namespace GUI {

TFGrayscaleAlphaPainter::TFGrayscaleAlphaPainter(QWidget* parent):
	TFAbstractPainter(parent),
	activeView_(ACTIVE_GRAYSCALE){
}

TFGrayscaleAlphaPainter::~TFGrayscaleAlphaPainter(){}

void TFGrayscaleAlphaPainter::correctView_(){

	paintAreaWidth_ = width() - 2*margin_;
	paintAreaHeight_ = height() - 2*margin_	- margin_ - colorBarSize_;

	view_ = TFColorMapPtr(new TFColorMap(paintAreaWidth_));
}

void TFGrayscaleAlphaPainter::paintEvent(QPaintEvent*){

	QPainter drawer(this);
	paintBackground_(&drawer, rect());
	paintCurves_(&drawer);
}

void TFGrayscaleAlphaPainter::paintCurves_(QPainter *drawer){

	int xBegin = margin_;
	int yBegin = margin_ + paintAreaHeight_;
	TFPaintingPoint origin(xBegin, yBegin);

	for(TFSize i = 0; i < paintAreaWidth_ - 2; ++i)
	{
		//alpha
		drawer->setPen(Qt::yellow);
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].alpha*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].alpha*paintAreaHeight_);
		//gray
		drawer->setPen(Qt::lightGray);
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].component1*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].component1*paintAreaHeight_);
	}
}

void TFGrayscaleAlphaPainter::mousePressEvent(QMouseEvent *e){

	if(e->button() == Qt::RightButton)
	{
		switch(activeView_)
		{
			case ACTIVE_GRAYSCALE:
			{
				activeView_ = ACTIVE_ALPHA;
				break;
			}
			case ACTIVE_ALPHA:
			{
				activeView_ = ACTIVE_GRAYSCALE;
				break;
			}
		}
		return;
	}

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
	if( mousePosition.y > (int)(paintAreaHeight_ + 2*margin_) )
	{
		return;
	}
	drawHelper_ = new TFPaintingPoint(mousePosition.x,
		(paintAreaHeight_ + 2*margin_) - mousePosition.y);
}

void TFGrayscaleAlphaPainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFGrayscaleAlphaPainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), (paintAreaHeight_ + 2*margin_) - e->pos().y());
		
	TFPaintingPoint begin = correctCoords_(*drawHelper_);
	TFPaintingPoint end = correctCoords_(mousePosition);

	addLine_(begin, end);

	*drawHelper_ = mousePosition;
	
	if(changed()) repaint();
}

void TFGrayscaleAlphaPainter::addPoint_(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight_;
	
	switch(activeView_)
	{
		case ACTIVE_GRAYSCALE:
		{
			(*view_)[point.x].component1 = yValue;
			(*view_)[point.x].component2 = yValue;
			(*view_)[point.x].component3 = yValue;
			break;
		}
		case ACTIVE_ALPHA:
		{
			(*view_)[point.x].alpha = yValue;
			break;
		}
	}
	changed_ = true;
}

} // namespace GUI
} // namespace M4D

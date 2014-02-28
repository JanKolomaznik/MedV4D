#include "MedV4D/GUI/TF/GrayscaleAlphaPainter1D.h"

namespace M4D {
namespace GUI {

GrayscaleAlphaPainter1D::GrayscaleAlphaPainter1D():
	Painter1D(Qt::white, Qt::white, Qt::white){

	componentNames_.push_back("Gray");
	componentNames_.push_back("Opacity");
}


GrayscaleAlphaPainter1D::GrayscaleAlphaPainter1D(const QColor& gray,
												 const QColor& alpha):
	Painter1D(gray, gray, gray, alpha){

	componentNames_.push_back("Gray");
	componentNames_.push_back("Opacity");
}

GrayscaleAlphaPainter1D::~GrayscaleAlphaPainter1D(){}

void GrayscaleAlphaPainter1D::updateFunctionView_(WorkCopy::Ptr workCopy){
		
	viewFunctionBuffer_ = QPixmap(area_.width(), area_.height());
	viewFunctionBuffer_.fill(noColor_);

	QPainter drawer(&viewFunctionBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	TF::Coordinates coordsBegin(1, 0);
	TF::Coordinates coordsEnd(1, 0);
	TF::Color colorBegin;
	TF::Color colorEnd;
	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		coordsBegin[0] = i;
		coordsEnd[0] = i + 1;
		colorBegin = workCopy->getColor(coordsBegin);
		colorEnd = workCopy->getColor(coordsEnd);

		x1 = origin.x + coordsBegin[0];
		x2 = origin.x + coordsEnd[0];

		y1 = origin.y + (1 - colorBegin.component1)*inputArea_.height();
		y2 = origin.y + (1 - colorEnd.component1)*inputArea_.height();
		drawer.setPen(component1_);
		drawer.drawLine(x1, y1,	x2, y2);

		y1 = origin.y + (1 - colorBegin.alpha)*inputArea_.height();
		y2 = origin.y + (1 - colorEnd.alpha)*inputArea_.height();
		drawer.setPen(alpha_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

} // namespace GUI
} // namespace M4D

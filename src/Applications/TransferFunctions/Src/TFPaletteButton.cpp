#include "TFPaletteButton.h"

#include <QtGui/QPainter>

namespace M4D {
namespace GUI {

TFPaletteButton::TFPaletteButton(QWidget* parent, const TFSize index):
	QWidget(parent),
	index_(index),
	active_(false),
	size_(128){
}

TFPaletteButton::~TFPaletteButton(){}

void TFPaletteButton::setup(){

	QSize fixedSize(size_, size_);
	resize(fixedSize);
	setMinimumSize(fixedSize);
	setMaximumSize(fixedSize);
}

void TFPaletteButton::activate(){

	active_ = true;
	repaint();
}

void TFPaletteButton::deactivate(){

	active_ = false;
	repaint();
}

void TFPaletteButton::paintEvent(QPaintEvent*){

	QPainter drawer(this);

	drawer.fillRect(rect(), QBrush(Qt::black));
	drawer.setPen(Qt::white);
	drawer.drawText(rect(), Qt::AlignCenter, QObject::tr(convert<TFSize, std::string>(index_).c_str()));

	if(active_)
	{
		drawBorder_(&drawer, Qt::red, 2);
	}
}

void TFPaletteButton::drawBorder_(QPainter* drawer, QColor color, int brushWidth){
	
	drawer->setPen( QPen(color, brushWidth) );

	drawer->drawLine(0, brushWidth - 1, width(), brushWidth - 1);
	drawer->drawLine(0, height() - (brushWidth - 1), width(), height() - (brushWidth - 1));
	drawer->drawLine(brushWidth - 1, 0, brushWidth - 1, height());
	drawer->drawLine(width() - (brushWidth - 1), 0, width() - (brushWidth - 1), height());
}

void TFPaletteButton::mouseReleaseEvent(QMouseEvent *){

	emit Triggered();
}

} // namespace GUI
} // namespace M4D
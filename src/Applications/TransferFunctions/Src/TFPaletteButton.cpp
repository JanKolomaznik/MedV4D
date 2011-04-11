#include "TFPaletteButton.h"

namespace M4D {
namespace GUI {

TFPaletteButton::TFPaletteButton(QWidget* parent, const TF::Size index):
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

	preview_ = QPixmap(fixedSize);
	preview_.fill(Qt::black);

	QPainter drawer(&preview_);

	drawer.setPen(Qt::white);
	drawer.drawText(rect(), Qt::AlignCenter,
		QString::fromStdString(TF::convert<TF::Size, std::string>(index_))
	);
	
	activePreview_ = QPixmap(fixedSize);
	activePreview_.fill(QColor(0,0,0,0));

	drawer.end();
	drawer.begin(&activePreview_);	

	drawer.setPen(QPen(Qt::red, 2));
	drawer.drawLine(0,0, width(), 0);
	drawer.drawLine(0, height(), width(), height());
	drawer.drawLine(0, 0, 0, height());
	drawer.drawLine(width(), 0, width(), height());
}

void TFPaletteButton::setPreview(const QPixmap& preview){

	preview_ = preview;
}

void TFPaletteButton::activate(){

	active_ = true;
	update();
}

void TFPaletteButton::deactivate(){

	active_ = false;
	update();
}

void TFPaletteButton::paintEvent(QPaintEvent*){

	QPainter drawer(this);

	drawer.drawPixmap(rect(), preview_);

	if(active_) drawer.drawPixmap(rect(), activePreview_);
}

void TFPaletteButton::mouseReleaseEvent(QMouseEvent *){

	if(!active_) emit Triggered(index_);
}

} // namespace GUI
} // namespace M4D
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
	drawer.end();
	
	activePreview_ = QPixmap(fixedSize);
	activePreview_.fill(QColor(0,0,0,0));

	drawer.begin(&activePreview_);	
	drawer.setPen(QPen(Qt::red, 2));
	drawer.drawLine(1,0, width(), 0);
	drawer.drawLine(1, height(), width(), height());
	drawer.drawLine(1, 0, 1, height());
	drawer.drawLine(width(), 1, width(), height());
	drawer.end();

	availablePreview_ = QPixmap(fixedSize);
	availablePreview_.fill(QColor(0,0,0,0));

	drawer.begin(&availablePreview_);
	drawer.setPen(QPen(Qt::blue, 2));
	drawer.drawLine(0,0, width(), 0);
	drawer.drawLine(0, height(), width(), height());
	drawer.drawLine(0, 0, 0, height());
	drawer.drawLine(width(), 0, width(), height());
}

void TFPaletteButton::setPreview(const QPixmap& preview){

	preview_ = preview;
}

void TFPaletteButton::setActive(const bool active){

	active_ = active;
	update();
}

void TFPaletteButton::setAvailable(const bool available){

	available_ = available;
	update();
}

void TFPaletteButton::paintEvent(QPaintEvent*){

	QPainter drawer(this);

	drawer.drawPixmap(rect(), preview_);

	if(active_) drawer.drawPixmap(rect(), activePreview_);
	else if(available_) drawer.drawPixmap(rect(), availablePreview_);
}

void TFPaletteButton::mouseReleaseEvent(QMouseEvent *){

	if(available_ && !active_) emit Triggered(index_);
}

} // namespace GUI
} // namespace M4D
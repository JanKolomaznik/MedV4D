#include "TFPaletteButton.h"

namespace M4D {
namespace GUI {

TFPaletteButton::TFPaletteButton(const TF::Size index, QWidget* parent):
	QFrame(parent),
	name_(this),
	index_(index),
	active_(false),
	available_(false),
	previewEnabled_(true),
	previewRect_(frameLineWidth_, frameLineWidth_ + nameHeight_, previewWidth, previewHeight){
}

TFPaletteButton::~TFPaletteButton(){}

void TFPaletteButton::setup(const std::string& name, bool enablePreview){

	togglePreview(enablePreview);

	//TODO frame nefunguje
	setFrameShape(QFrame::Panel);
	setFrameShadow(QFrame::Raised);
	setLineWidth(frameLineWidth_);
	setMidLineWidth(0);

	name_.setGeometry(frameLineWidth_, frameLineWidth_, previewWidth, nameHeight_);
	name_.setText(QString::fromStdString(name));
	name_.setAlignment(Qt::AlignCenter);

	QSize previewSize(previewWidth, previewHeight);
	QRect borderRect(0, 0, previewWidth, previewHeight);

	QPainter drawer;

	//---default-preview---
	preview_ = QImage(previewSize, QImage::Format_RGB16);
	preview_.fill(Qt::black);

	drawer.begin(&preview_);
	drawer.setPen(Qt::white);
	drawer.drawText(borderRect, Qt::AlignCenter, "No preview available.");
	drawer.end();
	//------
	
	activePreview_ = QPixmap(previewSize);
	activePreview_.fill(QColor(0,0,0,0));

	drawer.begin(&activePreview_);	
	drawer.setPen(QPen(Qt::red, 2));
	drawer.drawRect(borderRect);
	drawer.end();

	availablePreview_ = QPixmap(previewSize);
	availablePreview_.fill(QColor(0,0,0,0));

	drawer.begin(&availablePreview_);
	drawer.setPen(QPen(Qt::yellow, 2));
	drawer.drawRect(borderRect);

	show();
}

void TFPaletteButton::setName(const std::string& name){

	QString newName = QString::fromStdString(name);
	if(name_.text() != newName) name_.setText(newName);
}

void TFPaletteButton::setPreview(const QImage& preview){

	preview_ = preview;
	update();
}

QImage TFPaletteButton::getPreview(){

	return preview_;
}

void TFPaletteButton::togglePreview(bool enabled){
	
	previewEnabled_ = enabled;

	QSize buttonSize(
		2*frameLineWidth_ + previewWidth,
		2*frameLineWidth_ + nameHeight_
	);
	if(previewEnabled_) buttonSize.setHeight(buttonSize.height() + previewHeight);

	setMinimumSize(buttonSize);
	setMaximumSize(buttonSize);
	resize(buttonSize);
}

void TFPaletteButton::setActive(const bool active){

	active_ = active;
	if(active_) setAvailable(true);

	QFont changedFont = name_.font();
	changedFont.setBold(active);
	name_.setFont(changedFont);

	update();
}

void TFPaletteButton::setAvailable(const bool available){

	available_ = available;
	if(!available_) setActive(false);

	QFont changedFont = name_.font();
	changedFont.setItalic(!available);
	name_.setFont(changedFont);

	update();
}

bool TFPaletteButton::isActive(){

	return active_;
}

bool TFPaletteButton::isAvailable(){

	return available_;
}

void TFPaletteButton::paintEvent(QPaintEvent* e){

	QFrame::paintEvent(e);
	
	if(!previewEnabled_) return;

	QPainter drawer(this);

	drawer.drawImage(previewRect_, preview_);

	if(active_) drawer.drawPixmap(previewRect_, activePreview_);
	else if(available_) drawer.drawPixmap(previewRect_, availablePreview_);
}

void TFPaletteButton::mouseReleaseEvent(QMouseEvent *){

	if(available_) emit Triggered(index_);
}

//---Check---

TFPaletteCheckButton::TFPaletteCheckButton(const TF::Size index, QWidget* parent):
	TFPaletteButton(index, parent){
	
	previewRect_ = QRect(
		frameLineWidth_ + checkWidth_ + checkIndent_,
		frameLineWidth_ + nameHeight_,
		previewWidth,
		previewHeight
	);
}

TFPaletteCheckButton::~TFPaletteCheckButton(){}

void TFPaletteCheckButton::setup(const std::string& name, bool enablePreview){

	TFPaletteButton::setup(name, enablePreview);

	name_.setGeometry(frameLineWidth_ + checkWidth_, frameLineWidth_, previewWidth, nameHeight_);

	check_.setText("");
	check_.resize(checkWidth_, nameHeight_);
	check_.setParent(this);
	check_.show();

	bool checkConnected = QObject::connect(&check_, SIGNAL(toggled(bool)), this, SLOT(check_toggled(bool)));
	tfAssert(checkConnected);
}

void TFPaletteCheckButton::togglePreview(bool enabled){
	
	previewEnabled_ = enabled;

	QSize buttonSize(
		2*frameLineWidth_ + previewWidth + checkWidth_ + checkIndent_,
		nameHeight_ + 2*frameLineWidth_
	);
	if(previewEnabled_)
	{
		buttonSize.setHeight(buttonSize.height() + previewHeight);
		check_.move(checkIndent_, (buttonSize.height() - check_.height())/2);
	}
	else
	{
		check_.move(checkIndent_, 0);
	}

	setMinimumSize(buttonSize);
	setMaximumSize(buttonSize);
	resize(buttonSize);
}

void TFPaletteCheckButton::setActive(const bool active){

	TFPaletteButton::setActive(active);
	check_.setChecked(active);
}

void TFPaletteCheckButton::setAvailable(const bool available){

	TFPaletteButton::setAvailable(available);
	check_.setEnabled(available);
}

void TFPaletteCheckButton::check_toggled(bool toggled){

	if(available_ && (toggled != active_)) emit Triggered(index_);
}

} // namespace GUI
} // namespace M4D
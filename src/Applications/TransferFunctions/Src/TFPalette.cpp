#include "TFPalette.h"

namespace M4D {
namespace GUI {

TFPalette::TFPalette(QMainWindow* parent):
	QMainWindow(parent),
	ui_(new Ui::TFPalette),
	mainWindow_(parent),
	domain_(TFAbstractFunction<1>::defaultDomain),
	activeEditor_(-1),
	activeChanged_(false),
	creator_(parent, domain_){

    ui_->setupUi(this);
	setWindowTitle("Transfer Functions Palette");

	layout_ = new QVBoxLayout;
	layout_->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	ui_->scrollAreaWidget->setLayout(layout_);

	ui_->removeButton->setEnabled(false);
}

TFPalette::~TFPalette(){}

void TFPalette::setupDefault(){
	//default palette functions
}

/*
TF::MultiDColor<dim>::Map::Ptr TFPalette::getColorMap(){

	if(activeEditor_ < 0) on_actionNew_triggered();
	if(activeEditor_ < 0) exit(0);

	return palette_.find(activeEditor_)->second.holder->getColorMap();
}
*/
void TFPalette::setDomain(const TF::Size domain){

	if(domain_ == domain) return;
	domain_ = domain;

	creator_.setDomain(domain_);
	
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second.holder->setDomain(domain_);
	}
}

bool TFPalette::setHistogram(TF::Histogram::Ptr histogram, bool adjustDomain){

	histogram_ = histogram;
	
	if(adjustDomain)
	{
		domain_ = histogram_->size();
		creator_.setDomain(domain_);
	}
	else if(domain_ != histogram_->size())
	{
		return false;
	}

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second.holder->setHistogram(histogram_);
	}
	return true;
}

TF::Size TFPalette::getDomain(){

	return domain_;
}

M4D::Common::TimeStamp TFPalette::getTimeStamp(){

	if(palette_.empty())
	{
		++lastChange_;
		return lastChange_;
	}
	if(activeChanged_ || palette_.find(activeEditor_)->second.holder->changed()) ++lastChange_;

	activeChanged_ = false;

	return lastChange_;
}

void TFPalette::addToPalette_(TFAbstractHolder* holder){
	
	TF::Size addedIndex = indexer_.getIndex();

	holder->setup(mainWindow_, addedIndex);
	holder->setHistogram(histogram_);

	bool activateConnected = QObject::connect( holder, SIGNAL(Activate(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(activateConnected);
	bool closeConnected = QObject::connect( holder, SIGNAL(Close(TF::Size)), this, SLOT(close_triggered(TF::Size)));
	tfAssert(closeConnected);
	
	TFPaletteButton* button = new TFPaletteButton(ui_->scrollAreaWidget, addedIndex);
	button->setup();
	layout_->addWidget(button);
	button->show();

	bool buttonConnected = QObject::connect(button, SIGNAL(Triggered(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(buttonConnected);
	
	palette_.insert(std::make_pair<TF::Size, Editor>(addedIndex, Editor(holder, button)));
	
	QDockWidget* dockHolder = holder->getDockWidget();
	dockHolder->setFeatures(QDockWidget::AllDockWidgetFeatures);
	mainWindow_->addDockWidget(Qt::BottomDockWidgetArea, dockHolder);
	
	change_activeHolder(addedIndex);

	ui_->removeButton->setEnabled(true);
}

void TFPalette::removeFromPalette_(const TF::Size index){

	HolderMapIt toRemoveIt = palette_.find(index);
	if(index == activeEditor_) activateNext_(toRemoveIt);

	mainWindow_->removeDockWidget(toRemoveIt->second.holder->getDockWidget());
	delete toRemoveIt->second.holder;

	layout_->removeWidget(toRemoveIt->second.button);
	delete toRemoveIt->second.button;

	palette_.erase(toRemoveIt);
	indexer_.releaseIndex(index);

	if(palette_.empty()) ui_->removeButton->setEnabled(false);
}

void TFPalette::activateNext_(HolderMapIt it){
	
	if(palette_.size() == 1)
	{
		activeEditor_ = -1;
	}
	else
	{
		HolderMapIt nextActiveIt = it;
		++nextActiveIt;

		if(nextActiveIt == palette_.end())
		{
			nextActiveIt = it;
			--nextActiveIt;
		}

		change_activeHolder(nextActiveIt->second.holder->getIndex());
	}
}

void TFPalette::resizeEvent(QResizeEvent* e){

	ui_->paletteLayoutWidget->setGeometry(ui_->paletteArea->geometry());
}

void TFPalette::close_triggered(TF::Size index){

	removeFromPalette_(index);
}

void TFPalette::change_activeHolder(TF::Size index){

	if(index == activeEditor_) return;

	Editor active;
	if(activeEditor_ >= 0)
	{
		active = palette_.find(activeEditor_)->second;
		active.button->deactivate();
		active.holder->deactivate();
	}

	activeEditor_ = index;
	active = palette_.find(activeEditor_)->second;
	active.button->activate();
	active.holder->activate();

	activeChanged_ = true;
}

void TFPalette::on_addButton_clicked(){

	TFAbstractHolder* created = creator_.createTransferFunction();

	if(!created) return;
	
	addToPalette_(created);
}

void TFPalette::on_removeButton_clicked(){

	palette_.find(activeEditor_)->second.holder->close();
}

void TFPalette::closeEvent(QCloseEvent *e){
	
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second.holder->close();
	}

	e->accept();
}

} // namespace GUI
} // namespace M4D

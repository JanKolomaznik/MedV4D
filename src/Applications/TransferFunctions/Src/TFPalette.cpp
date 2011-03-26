#include "TFPalette.h"

namespace M4D {
namespace GUI {

TFPalette::TFPalette(QMainWindow* parent):
	QMainWindow(parent),
	ui_(new Ui::TFPalette),
	mainWindow_(parent),
	domain_(TFAbstractFunction<1>::defaultDomain),
	activeHolder_(-1),
	activeChanged_(false),
	creator_(parent, domain_){

    ui_->setupUi(this);
	setWindowTitle("Transfer Functions Palette");

	layout_ = new QVBoxLayout;
	layout_->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
	ui_->scrollAreaWidget->setLayout(layout_);
}

TFPalette::~TFPalette(){}
/*
void TFPalette::setupDefault(){
	
	domain_ = TFApplyFunctionInterface::defaultDomain;
	//newTF_triggered(TF::Types::PredefinedCustom);
}
*/
/*
TF::MultiDColor<dim>::Map::Ptr TFPalette::getColorMap(){

	if(activeHolder_ < 0) on_actionNew_triggered();
	if(activeHolder_ < 0) exit(0);

	return palette_.find(activeHolder_)->second->getColorMap();
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
		it->second->setDomain(domain_);
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
		it->second->setHistogram(histogram_);
	}
	return true;
}

TF::Size TFPalette::getDomain(){

	return domain_;
}

M4D::Common::TimeStamp TFPalette::getTimeStamp(/*bool& noFunctionAvailable*/){

	if(palette_.empty())
	{
		++lastChange_;
		return lastChange_;
	}
	if(activeChanged_ || palette_.find(activeHolder_)->second->changed()) ++lastChange_;

	activeChanged_ = false;

	return lastChange_;
}

void TFPalette::addToPalette_(TFHolderInterface* holder){
	
	TF::Size addedIndex = indexer_.getIndex();

	holder->setup(addedIndex);
	holder->setHistogram(histogram_);
	palette_.insert(std::make_pair<TF::Size, TFHolderInterface*>(addedIndex, holder));
	holder->createPaletteButton(ui_->scrollAreaWidget);

	TFPaletteButton* addedButton = holder->getButton();
	layout_->addWidget(addedButton);
	addedButton->show();

	holder->connectToTFPalette(this);

	holder->createDockWidget(mainWindow_);	
	mainWindow_->addDockWidget(Qt::BottomDockWidgetArea, holder->getDockWidget());	
	
	change_activeHolder(addedIndex);
}

void TFPalette::removeFromPalette_(const TF::Size index){

	bool last = palette_.size() == 1;

	HolderMapIt toRemoveIt = palette_.find(index);
	mainWindow_->removeDockWidget(toRemoveIt->second->getDockWidget());

	if(index == activeHolder_)
	{
		if(last)
		{
			activeHolder_ = -1;
		}
		else
		{
			HolderMapIt nextActiveIt = toRemoveIt;
			++nextActiveIt;
			if(nextActiveIt == palette_.end())
			{
				nextActiveIt = toRemoveIt;
				--nextActiveIt;
			}
			TF::Size toActivate = nextActiveIt->second->getIndex();
			change_activeHolder(toActivate);
		}
	}

	TFPaletteButton* toRemoveButton = toRemoveIt->second->getButton();
	layout_->removeWidget(toRemoveButton);
	delete toRemoveButton;
	delete toRemoveIt->second;
	palette_.erase(toRemoveIt);
	indexer_.releaseIndex(index);
}

void TFPalette::resizeEvent(QResizeEvent* e){

	QMainWindow::resizeEvent(e);

	ui_->scrollArea->setGeometry(ui_->paletteArea->rect());
}

void TFPalette::close_triggered(TF::Size index){

	//if(palette_.size() <= 1) exit(0);

	removeFromPalette_(index);
}

void TFPalette::on_actionLoad_triggered(){

	TFHolderInterface* loaded = creator_.loadTransferFunction();

	if(!loaded) return;
	
	addToPalette_(loaded);
}

void TFPalette::on_actionNew_triggered(){

	TFHolderInterface* created = creator_.createTransferFunction();

	if(!created) return;
	
	addToPalette_(created);
}

void TFPalette::change_activeHolder(TF::Size index){

	if(index == activeHolder_) return;

	if(activeHolder_ >= 0) palette_.find(activeHolder_)->second->deactivate();

	activeHolder_ = index;
	palette_.find(activeHolder_)->second->activate();
	activeChanged_ = true;
}

//---Indexer---

TFPalette::Indexer::Indexer(): nextIndex_(0){}

TFPalette::Indexer::~Indexer(){}

TF::Size TFPalette::Indexer::getIndex(){

	TF::Size index = nextIndex_;
	if(!released_.empty())
	{
		index = released_[released_.size()-1];
		released_.pop_back();
	}
	else
	{
		++nextIndex_;
	}
	return index;
}

void TFPalette::Indexer::releaseIndex(const TF::Size index){

	if(index == (nextIndex_-1))
	{
		--nextIndex_;

		IndexesIt newBegin = released_.begin();
		IndexesIt end = released_.end();
		while( (newBegin != end) && (*newBegin == (nextIndex_-1)) )
		{
			++newBegin;
			--nextIndex_;
		}
		if(newBegin == end) released_.clear();
		else released_ = Indexes(newBegin, end);
	}
	else
	{
		IndexesIt it = released_.begin();
		IndexesIt end = released_.end();

		while( (it != end) && (*it > index) ) ++it;
		released_.insert(it, index);
	}
}

//------

} // namespace GUI
} // namespace M4D

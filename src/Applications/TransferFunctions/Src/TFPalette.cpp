#include "TFPalette.h"

namespace M4D {
namespace GUI {

TFPalette::TFPalette(QMainWindow* parent):
	QMainWindow(parent),
	ui_(new Ui::TFPalette),
	mainWindow_(parent),
	domain_(TFAbstractFunction::defaultDomain),
	activeHolder_(-1){

    ui_->setupUi(this);
	setWindowTitle("Transfer Functions Palette");

	layout_ = new QVBoxLayout;
	layout_->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
	ui_->scrollAreaWidget->setLayout(layout_);
}

TFPalette::~TFPalette(){}
/*
void TFPalette::setupDefault(){
	
	domain_ = TFAbstractFunction::defaultDomain;
	//newTF_triggered(TF::Types::PredefinedCustom);
}
*/
void TFPalette::setDomain(TF::Size domain){

	if(domain_ == domain) return;
	domain_ = domain;
	
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second->setDomain(domain_);
	}
}

TF::Size TFPalette::getDomain(){

	return domain_;
}

M4D::Common::TimeStamp TFPalette::getTimeStamp(/*bool& noFunctionAvailable*/){

	//noFunctionAvailable = false;
	if(palette_.empty())
	{
		//noFunctionAvailable = true;
		++lastChange_;
		return lastChange_;
	}
	if(palette_.find(activeHolder_)->second->changed()) ++lastChange_;

	return lastChange_;
}

void TFPalette::addToPalette_(TFHolder* holder){
	
	TF::Size addedIndex = indexer_.getIndex();

	holder->setup(addedIndex);
	holder->setHistogram(histogram_);
	palette_.insert(std::make_pair<TF::Size, TFHolder*>(addedIndex, holder));
	holder->createPaletteButton(ui_->scrollAreaWidget);

	TFPaletteButton* addedButton = holder->getButton();
	layout_->addWidget(addedButton);
	addedButton->show();

	holder->connectToTFPalette(this);

	holder->createDockWidget(mainWindow_);	
	mainWindow_->addDockWidget(Qt::BottomDockWidgetArea, holder->getDockWidget());	
	
	change_activeHolder(addedIndex);
}

void TFPalette::removeFromPalette_(TF::Size index){

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

	TFHolder* loaded = TFCreator::loadTransferFunction(this, domain_);

	if(!loaded) return;
	
	addToPalette_(loaded);
}

void TFPalette::on_actionNew_triggered(){

	TFHolder* created = TFCreator::createTransferFunction(this, domain_);

	if(!created) return;
	
	addToPalette_(created);
}

void TFPalette::change_activeHolder(TF::Size index){

	if(activeHolder_ >= 0) palette_.find(activeHolder_)->second->deactivate();

	activeHolder_ = index;

	palette_.find(activeHolder_)->second->activate();
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

void TFPalette::Indexer::releaseIndex(TF::Size index){

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

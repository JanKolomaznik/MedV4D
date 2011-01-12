#include "TFPalette.h"

namespace M4D {
namespace GUI {

TFPalette::TFPalette(QMainWindow* parent, TFSize domain):
	QMainWindow(parent),
	ui_(new Ui::TFPalette),
	mainWindow_(parent),
	activeHolder_(-1),
	domain_(domain){

    ui_->setupUi(this);
	setWindowTitle("Transfer Functions Palette");

	tfActions_ = TFHolderFactory::createMenuTFActions(ui_->menuNew);
	connectTFActions_();
}

TFPalette::~TFPalette(){
	/*
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		if(it->second) delete it->second;
	}
	*/
}

M4D::Common::TimeStamp TFPalette::getTimeStamp(){

	return palette_.find(activeHolder_)->second->getLastChangeTime();
}

bool TFPalette::connectTFActions_(){

	bool allConnected = true;

	const std::string shortcutBase = "Shift+F";
	TFSize actionCounter = 1;
	TFActionsIt begin = tfActions_.begin();
	TFActionsIt end = tfActions_.end();
	for(TFActionsIt it = begin; it!=end; ++it)
	{
		bool menuActionConnected = QObject::connect( *it, SIGNAL(TFActionClicked(TFHolder::Type)), this, SLOT(newTF_triggered(TFHolder::Type)));
		tfAssert(menuActionConnected);

		if(menuActionConnected)
		{
			ui_->menuNew->addAction(*it);

			std::string strShortcut = shortcutBase;
			strShortcut.append( convert<TFSize, std::string>(actionCounter) );
			(*it)->setShortcut( QKeySequence(QObject::tr(strShortcut.c_str())) );

			++actionCounter;
		}
		allConnected = allConnected && menuActionConnected;
	}

	return allConnected;
}

void TFPalette::setupDefault(){
	
	newTF_triggered(TFHolder::TFHolderGrayscale);
}

void TFPalette::setHistogram(HistogramPtr histogram){

	//TODO compute ColorMapPtr from histogram and set it to all holders
}

void TFPalette::addToPalette_(TFHolder* holder){
	
	TFSize addedIndex = indexer_.getIndex();

	holder->setUp(addedIndex);
	palette_.insert(std::make_pair<TFSize, TFHolder*>(addedIndex, holder));
	holder->createPaletteButton(ui_->layoutWidget);

	TFPaletteButton* addedButton = holder->getButton();
	ui_->paletteLayout->addWidget(addedButton);
	addedButton->show();

	holder->connectToTFPalette(this);

	holder->createDockWidget(mainWindow_);	
	mainWindow_->addDockWidget(Qt::BottomDockWidgetArea, holder->getDockWidget());	
	
	change_activeHolder(addedIndex);
}

void TFPalette::removeFromPalette_(TFSize index){

	tfAssert(palette_.size() > 1);

	HolderMapIt toRemoveIt = palette_.find(index);
	mainWindow_->removeDockWidget(toRemoveIt->second->getDockWidget());

	if(index == activeHolder_)
	{
		HolderMapIt nextActiveIt = toRemoveIt;
		++nextActiveIt;
		if(nextActiveIt == palette_.end())
		{
			nextActiveIt = toRemoveIt;
			--nextActiveIt;
		}
		TFSize toActivate = nextActiveIt->second->getIndex();
		change_activeHolder(toActivate);	
	}

	TFPaletteButton* toRemoveButton = toRemoveIt->second->getButton();
	ui_->paletteLayout->removeWidget(toRemoveButton);
	delete toRemoveButton;
	delete toRemoveIt->second;
	palette_.erase(toRemoveIt);
	indexer_.releaseIndex(index);
}

void TFPalette::resizeEvent(QResizeEvent* e){

	QMainWindow::resizeEvent(e);

	ui_->layoutWidget->setGeometry(ui_->paletteArea->rect());
}

void TFPalette::close_triggered(TFSize index){

	if(palette_.size() <= 1) exit(0);

	removeFromPalette_(index);
}
/*
void TFPalette::save_triggered(){

	palette_.find(activeHolder_)->second->save();
}
*/
void TFPalette::on_actionLoad_triggered(){

	TFHolder* loaded = TFHolderFactory::loadHolder(this, domain_);
	if(!loaded) return;
	
	addToPalette_(loaded);
}

void TFPalette::newTF_triggered(TFHolder::Type tfType){

	TFHolder* holder = TFHolderFactory::createHolder(this, tfType, domain_);

	if(!holder){
		QMessageBox::warning(this, QObject::tr("Transfer Functions"), QObject::tr("Creating error."));
		return;
	}

	addToPalette_(holder);
}

void TFPalette::change_activeHolder(TFSize index){

	if(activeHolder_ >= 0) palette_.find(activeHolder_)->second->getButton()->deactivate();

	activeHolder_ = index;

	palette_.find(activeHolder_)->second->getButton()->activate();
}

//---Indexer---

TFPalette::Indexer::Indexer(): nextIndex_(0){}

TFPalette::Indexer::~Indexer(){}

TFSize TFPalette::Indexer::getIndex(){

	TFSize index = nextIndex_;
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

void TFPalette::Indexer::releaseIndex(TFSize index){

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

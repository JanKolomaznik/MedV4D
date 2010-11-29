#include "TFWindow.h"

namespace M4D {
namespace GUI {

TFWindow::TFWindow(QMainWindow* parent):
	QWidget(parent),
	ui_(new Ui::TFWindow),
	mainWindow_(parent),
	activeHolder_(-1),
	holderInWindow_(-1){

    ui_->setupUi(this);
	ui_->paletteArea->hide();
}

TFWindow::~TFWindow(){

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		delete it->second;
	}

	DockHolderMapIt beginReleased = releasedHolders_.begin();
	DockHolderMapIt endReleased = releasedHolders_.end();
	for(DockHolderMapIt it = beginReleased; it != endReleased; ++it)
	{
		delete it->second;
	}
}

void TFWindow::createMenu(QMenuBar* menubar){

	menuTF_ = new QMenu(QString::fromStdString("Transfer Function"), menubar);

	//menu new
	menuNew_ = new QMenu(QString::fromStdString("New"), menuTF_);	
	tfActions_ = TFHolderFactory::createMenuTFActions(menuNew_);

	TFActionsIt begin = tfActions_.begin();
	TFActionsIt end = tfActions_.end();
	for(TFActionsIt it = begin; it!=end; ++it)
	{
		bool menuActionConnected = QObject::connect( *it, SIGNAL(TFActionClicked(const TFHolderType&)), this, SLOT(newTF_triggered(const TFHolderType&)));
		tfAssert(menuActionConnected);
		menuNew_->addAction(*it);
	}

	menuTF_->addAction(menuNew_->menuAction());

	//action load
	actionLoad_ = new QAction(QString::fromStdString("Load"), menuTF_);
	bool loadConnected = QObject::connect( actionLoad_, SIGNAL(triggered()), this, SLOT(load_triggered()));
	tfAssert(loadConnected);
	actionLoad_->setShortcut(QKeySequence(QObject::tr("Ctrl+L")));
	menuTF_->addAction(actionLoad_);

	//action save
	actionSave_ = new QAction(QString::fromStdString("Save"), menuTF_);
	actionSave_->setEnabled(false);
	bool saveConnected = QObject::connect( actionSave_, SIGNAL(triggered()), this, SLOT(save_triggered()));
	tfAssert(saveConnected);
	actionSave_->setShortcut(QKeySequence::Save);
	menuTF_->addAction(actionSave_);

	//separator
	menuTF_->addSeparator();	

	//action exit
	actionExit_ = new QAction(QString::fromStdString("Close"), menuTF_);
	bool exitConnected = QObject::connect( actionExit_, SIGNAL(triggered()), this, SLOT(close_triggered()));
	tfAssert(exitConnected);
	actionExit_->setShortcut(QKeySequence::Close);
	menuTF_->addAction(actionExit_);
	
	menuTF_->setEnabled(true);
	menubar->addAction(menuTF_->menuAction());
}

void TFWindow::setupDefault(){
	
	newTF_triggered(TFHOLDER_GRAYSCALE);
}

void TFWindow::addToPalette_(TFAbstractHolder* holder){
	
	TFSize addedIndex = indexer_.getIndex();

	holder->setUp(addedIndex);
	palette_.insert(std::make_pair<TFSize, TFAbstractHolder*>(addedIndex, holder));
	holder->createPaletteButton(ui_->paletteArea);
	ui_->paletteLayout->addWidget(holder->getButton());

	inWindowHolders_.insert(std::make_pair<TFSize, TFAbstractHolder*>(addedIndex, holder));
	if(inWindowHolders_.size() == 2) ui_->paletteArea->show();	//shows buttons with more TFs in window

	holder->connectToTFWindow(this);

	changeHolderInWindow_(addedIndex, true);
	change_activeHolder(addedIndex);
}

void TFWindow::removeFromPalette_(){

	TFSize toRemoveIndex = activeHolder_;
	TFAbstractHolder* toRemove = palette_.find(toRemoveIndex)->second;
	TFDockHolder* dockToRemove = NULL;

	bool wasReleased = toRemove->isReleased();	
	if(wasReleased)
	{
		DockHolderMapIt dockPairIt = releasedHolders_.find(toRemoveIndex);
		dockToRemove = dockPairIt->second;
		releasedHolders_.erase(dockPairIt);
	}
	else
	{
		ui_->paletteLayout->removeWidget(toRemove->getButton());
		inWindowHolders_.erase(toRemoveIndex);
		if(inWindowHolders_.size() == 1) ui_->paletteArea->hide();	//hides buttons with only 1 TF in window
	}

	indexer_.releaseIndex(toRemoveIndex);
	TFSize toActivate = getFirstInWindow_();
	changeHolderInWindow_(toActivate, false);
	change_activeHolder(toActivate);	
	
	palette_.erase(toRemoveIndex);
	if(wasReleased)
	{
		delete dockToRemove;	//includes delete toRemove
	}
	else
	{
		delete toRemove;
	}
}

void TFWindow::mousePressEvent(QMouseEvent* e){

	change_activeHolder(holderInWindow_);
	QWidget::mousePressEvent(e);
}

void TFWindow::keyPressEvent(QKeyEvent* e){

	change_activeHolder(holderInWindow_);
	QWidget::keyPressEvent(e);
}

void TFWindow::resizeEvent(QResizeEvent*){

	ui_->holderArea->resize(width(), (3*height())/4); //holder takesa 3/4
	ui_->paletteArea->resize(width(), height() - ui_->holderArea->height());	//palette takes the rest
	ui_->paletteArea->move(0, ui_->holderArea->height());

	emit ResizeHolder(holderInWindow_, ui_->holderArea->rect());
}

void TFWindow::close_triggered(){

	if(palette_.size() <= 1) exit(0);

	removeFromPalette_();
}

void TFWindow::save_triggered(){

	palette_.find(activeHolder_)->second->save();
}

void TFWindow::load_triggered(){

	TFAbstractHolder* loaded = TFHolderFactory::loadHolder(ui_->holderArea);
	if(!loaded) return;
	
	addToPalette_(loaded);

	actionSave_->setEnabled(true);
}

void TFWindow::newTF_triggered(const TFHolderType &tfType){

	TFAbstractHolder* holder = TFHolderFactory::createHolder(ui_->holderArea, tfType);

	if(!holder){
		QMessageBox::warning(this, QObject::tr("Transfer Functions"), QObject::tr("Creating error."));
		return;
	}

	addToPalette_(holder);

	actionSave_->setEnabled(true);
}

void TFWindow::change_activeHolder(const TFSize& index){

	activeHolder_ = index;
	TFAbstractHolder* oldHolder = palette_.find(activeHolder_)->second;
	if(!oldHolder->isReleased()) changeHolderInWindow_(index, true);
}

void TFWindow::release_triggered(){
	
	TFAbstractHolder* toRelease = palette_.find(activeHolder_)->second;
	if(toRelease->isReleased())
	{
		toRelease->setParent(this);

		TFDockHolder* dock = (releasedHolders_.find(activeHolder_))->second;
		dock->setWidget(NULL);
		delete dock;
		releasedHolders_.erase(activeHolder_);

		if(inWindowHolders_.size() == 1) ui_->paletteArea->show();	//shows buttons with more TFs in window
		inWindowHolders_.insert(std::make_pair<TFSize, TFAbstractHolder*>(activeHolder_, toRelease));
		toRelease->getButton()->show();

		toRelease->setReleased(false);

		changeHolderInWindow_(activeHolder_, true);
	}
	else
	{
		if(inWindowHolders_.size() <= 1) return;	//cannt release last TF in window

		inWindowHolders_.erase(activeHolder_);
		toRelease->getButton()->hide();

		if(inWindowHolders_.size() == 1) ui_->paletteArea->hide();	//hides buttons with only 1 TF in window

		TFDockHolder* released = new TFDockHolder("Transfer Function " + activeHolder_, this, activeHolder_);
		released->setWidget(toRelease);
		released->connectSignals();

		mainWindow_->addDockWidget(Qt::TopDockWidgetArea, released);
		released->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);

		releasedHolders_.insert(std::make_pair<TFSize, TFDockHolder*>(activeHolder_, released));

		toRelease->setReleased(true);

		TFSize index = getFirstInWindow_();
		changeHolderInWindow_(index, false);
	}
}

TFSize TFWindow::getFirstInWindow_(){
		
	if(inWindowHolders_.empty())
	{
		TFSize active = activeHolder_;

		activeHolder_ = releasedHolders_.begin()->first;
		release_triggered();

		activeHolder_ = active;
	}
	return inWindowHolders_.begin()->first;
}

void TFWindow::changeHolderInWindow_(const TFSize& index, const bool& hideOld){

	if(index == holderInWindow_) return;

	tfAssert(index < palette_.size());

	if(hideOld && (holderInWindow_ >= 0)) palette_.find(holderInWindow_)->second->hide();
	holderInWindow_ = index;
	palette_.find(holderInWindow_)->second->show();	

	emit ResizeHolder(index, ui_->holderArea->rect());	
}


//---Indexer---

TFWindow::Indexer::Indexer(): nextIndex_(0){}

TFWindow::Indexer::~Indexer(){}

TFSize TFWindow::Indexer::getIndex(){

	if(!released_.empty()) return *released_.begin();
	return nextIndex_++;
}

void TFWindow::Indexer::releaseIndex(const TFSize& index){

	if(index == (nextIndex_-1))
	{
		--nextIndex_;
		while( (!released_.empty()) && (released_[released_.size()-1] == (nextIndex_-1)) ) --nextIndex_;
	}
	else
	{
		IndexesIt it = released_.begin();
		IndexesIt end = released_.end();
		while( (it < end) && (*it < index) ) ++it;
		released_.insert(it, index);
	}
}

//------

} // namespace GUI
} // namespace M4D

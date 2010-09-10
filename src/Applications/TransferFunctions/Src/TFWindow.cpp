#include "TFWindow.h"
#include "ui_TFWindow.h"

#include <cassert>

#include <TFSimpleHolder.h>


TFWindow::TFWindow(): ui_(new Ui::TFWindow), holder_(NULL){

    ui_->setupUi(this);
}

TFWindow::~TFWindow(){

	for(unsigned i = 0; i < tfActions_.size(); ++i)
	{
		delete tfActions_[i];
	}

	if(holder_) delete holder_;
    if(ui_) delete ui_;
}

void TFWindow::build(){

	tfActions_ = TFHolderFactory::createMenuTFActions(this, ui_->menuNew);

	for(unsigned i = 0; i < tfActions_.size(); ++i)
	{
		 bool ok = QObject::connect( tfActions_[i], SIGNAL(TFActionClicked(TFType&)), this, SLOT(newTF_triggered(TFType&)));
	}
}

void TFWindow::setupHolder(){

	holder_->setUp(this, QRect(0, MENU_SPACE, width(), height() - MENU_SPACE));
	QObject::connect( holder_, SIGNAL(UseTransferFunction(TFAbstractFunction&)), this, SLOT(modify_data(TFAbstractFunction&)));
	QObject::connect( this, SIGNAL(ResizeHolder(const QRect)), holder_, SLOT(size_changed(const QRect)));
}

void TFWindow::resizeEvent(QResizeEvent* e){

	ui_->menuBar->setGeometry(QRect(0, 0, width(), ui_->menuBar->height()));
	emit ResizeHolder(QRect(0, MENU_SPACE, width(), height() - MENU_SPACE));
}

void TFWindow::on_exit_triggered(){

    close();
}

void TFWindow::on_save_triggered(){

	holder_->save();
}

void TFWindow::on_load_triggered(){

	holder_ = TFHolderFactory::loadHolder(this);

	if(!holder_){
		QMessageBox::warning(this, QObject::tr("Transfer Functions"), QObject::tr("Loading error."));
		return;
	}

	ui_->save->setEnabled(true);
	setupHolder();
}

void TFWindow::newTF_triggered(TFType &tfType){

	if(holder_)
	{
		delete holder_;
		holder_ = NULL;
	}

	holder_ = TFHolderFactory::createHolder(tfType);

	if(!holder_){
		QMessageBox::warning(this, QObject::tr("Transfer Functions"), QObject::tr("Creating error."));
		return;
	}

	ui_->save->setEnabled(true);
	setupHolder();
}

void TFWindow::modify_data(TFAbstractFunction &transferFunction){

	emit AdjustByTransferFunction(transferFunction);
}

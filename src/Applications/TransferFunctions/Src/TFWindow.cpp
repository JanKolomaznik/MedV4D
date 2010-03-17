#include "TFWindow.h"
#include "ui_TFWindow.h"

#include <cassert>

#include <TFSimpleHolder.h>

/*
 * constructor, destructor
 */

TFWindow::TFWindow(): ui(new Ui::TFWindow), _holder(NULL){

    ui->setupUi(this);
}

TFWindow::~TFWindow(){

	if(_holder) delete _holder;
    if(ui) delete ui;
}

void TFWindow::build(){
}

void TFWindow::setupHolder(){

	_holder->setup(this, QRect(10,30,560,400));
	QObject::connect( _holder, SIGNAL(UseTransferFunction(TFAbstractFunction&)), this, SLOT(modify_data(TFAbstractFunction&)));
}

void TFWindow::on_exit_triggered(){

    close();
}

void TFWindow::on_save_triggered(){

	_holder->save();
}

void TFWindow::on_load_triggered(){

	_holder = TFHolderFactory::load(this);
	setupHolder();
}

void TFWindow::on_simple_triggered(){

	if(_holder)
	{
		delete _holder;
		_holder = NULL;
	}
	_holder = new TFSimpleHolder();
	setupHolder();
}

void TFWindow::modify_data(TFAbstractFunction &transferFunction){

	emit AdjustByTransferFunction(transferFunction);
}

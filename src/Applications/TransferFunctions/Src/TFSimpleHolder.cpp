
#include "TFSimpleHolder.h"

TFSimpleHolder::TFSimpleHolder(){

	_type = TFTYPE_SIMPLE;

	_basicTools->name->setText(QString::fromStdString(_function.name));

	_painter.setup(this, QRect(25, 70, FUNCTION_RANGE_SIMPLE + 20, COLOR_RANGE_SIMPLE + 20));
}

TFSimpleHolder::~TFSimpleHolder(){

	if(_basicTools) delete _basicTools;
}

void TFSimpleHolder::setup(QWidget *parent, const QRect rect){

    setGeometry(rect);	
	setParent(parent);
	show();
}

void TFSimpleHolder::_save(QFile &file){

	_function.setPoints(_painter.getView());

	_function.name = _basicTools->name->text().toStdString();

	 TFXmlSimpleWriter writer;
     writer.write(&file, _function);
}

bool TFSimpleHolder::_load(QFile &file){

	TFXmlSimpleReader reader;

	bool error = false;

	TFSimpleFunction loaded = reader.read(&file, error);

	if (error || reader.error())
	{
		return false;
	}

	_function = loaded;

	_painter.setView(_function.getPointMap());

	_basicTools->name->setText(QString::fromStdString(_function.name));

	return true;
}

void TFSimpleHolder::on_use_clicked(){

	_function.setPoints(_painter.getView());

	emit UseTransferFunction(_function);
}
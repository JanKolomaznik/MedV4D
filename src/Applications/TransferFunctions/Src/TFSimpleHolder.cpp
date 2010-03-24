
#include "TFSimpleHolder.h"

TFSimpleHolder::TFSimpleHolder(){

	_type = TFTYPE_SIMPLE;

	_basicTools->name->setText(QString::fromStdString(_function.name));
}

TFSimpleHolder::~TFSimpleHolder(){

	if(_basicTools) delete _basicTools;
}

void TFSimpleHolder::setup(QWidget *parent, const QRect rect){

	_painter.setup(this);
	size_changed(rect);
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

void TFSimpleHolder::size_changed(const QRect rect){

	setGeometry(rect);

	int newWidth = rect.width() - 2*PAINTER_X;
	int newHeight = rect.height() - 2*PAINTER_Y;

	_function.setPoints(_painter.getView());
	_function.recalculate(newWidth - 2*PAINTER_MARGIN_H, newHeight - 2*PAINTER_MARGIN_V);

	_painter.resize(QRect(PAINTER_X, PAINTER_Y, newWidth, newHeight));
	_painter.setView(_function.getPointMap());
}
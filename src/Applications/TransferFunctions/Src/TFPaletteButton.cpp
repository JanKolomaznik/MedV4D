#include "TFPaletteButton.h"

#include <QtGui/QKeySequence>

namespace M4D {
namespace GUI {

TFPaletteButton::TFPaletteButton(QWidget* parent, TFSize index):
	QPushButton(QString::number(index+1), parent),
	index_(index){

	bool buttonEnabled = QObject::connect( this, SIGNAL(clicked()), this, SLOT(button_triggered()));
	tfAssert(buttonEnabled);

	setCheckable(true);

	std::string shortcut = "F" + convert<TFSize,std::string>(index_+1);
	setShortcut( QKeySequence(QString::fromStdString(shortcut)) );
}

TFPaletteButton::~TFPaletteButton(){}

void TFPaletteButton::changeIndex(TFSize index){

	index_ = index;
	setText(QString::number(index_+1));

	std::string shortcut = "F" + convert<TFSize,std::string>(index_+1);
	setShortcut( QKeySequence(QString::fromStdString(shortcut)) );
}

void TFPaletteButton::button_triggered(){

	emit TFPaletteSignal(index_);
}
} // namespace GUI
} // namespace M4D
#include "TFRGBaPainter.h"

namespace M4D {
namespace GUI {

TFRGBaPainter::TFRGBaPainter():
	TFSimplePainter(Qt::red, Qt::green, Qt::blue){

	componentNames_.push_back("Red");
	componentNames_.push_back("Green");
	componentNames_.push_back("Blue");
	componentNames_.push_back("Opacity");
}


TFRGBaPainter::TFRGBaPainter(const QColor& red,
							 const QColor& green,
							 const QColor& blue,
							 const QColor& alpha):
	TFSimplePainter(red, green, blue, alpha){

	componentNames_.push_back("Red");
	componentNames_.push_back("Green");
	componentNames_.push_back("Blue");
	componentNames_.push_back("Opacity");
}

TFRGBaPainter::~TFRGBaPainter(){}

} // namespace GUI
} // namespace M4D

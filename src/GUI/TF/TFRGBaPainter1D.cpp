#include "GUI/TF/TFRGBaPainter1D.h"

namespace M4D {
namespace GUI {

TFRGBaPainter1D::TFRGBaPainter1D():
	TFPainter1D(Qt::red, Qt::green, Qt::blue){

	componentNames_.push_back("Red");
	componentNames_.push_back("Green");
	componentNames_.push_back("Blue");
	componentNames_.push_back("Opacity");
}


TFRGBaPainter1D::TFRGBaPainter1D(const QColor& red,
							 const QColor& green,
							 const QColor& blue,
							 const QColor& alpha):
	TFPainter1D(red, green, blue, alpha){

	componentNames_.push_back("Red");
	componentNames_.push_back("Green");
	componentNames_.push_back("Blue");
	componentNames_.push_back("Opacity");
}

TFRGBaPainter1D::~TFRGBaPainter1D(){}

} // namespace GUI
} // namespace M4D

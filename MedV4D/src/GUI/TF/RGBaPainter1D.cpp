#include "MedV4D/GUI/TF/RGBaPainter1D.h"

namespace M4D {
namespace GUI {

RGBaPainter1D::RGBaPainter1D():
	Painter1D(Qt::red, Qt::green, Qt::blue){

	componentNames_.push_back("Red");
	componentNames_.push_back("Green");
	componentNames_.push_back("Blue");
	componentNames_.push_back("Opacity");
}


RGBaPainter1D::RGBaPainter1D(const QColor& red,
							 const QColor& green,
							 const QColor& blue,
							 const QColor& alpha):
	Painter1D(red, green, blue, alpha){

	componentNames_.push_back("Red");
	componentNames_.push_back("Green");
	componentNames_.push_back("Blue");
	componentNames_.push_back("Opacity");
}

RGBaPainter1D::~RGBaPainter1D(){}

} // namespace GUI
} // namespace M4D

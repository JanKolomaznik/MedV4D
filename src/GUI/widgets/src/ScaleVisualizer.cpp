#include "GUI/widgets/ScaleVisualizer.h"

void
ScaleVisualizer::paintEvent( QPaintEvent * event )
{

}

QTransform
ScaleVisualizer::PrepareCanvasTransform( double aMin, double aMax, double aMMin, double aMMax, int aWidth, int aHeight )
{
	double width = aMax - aMin;
	double height = aMMax - aMMin;
	QTransform result = QTransform::fromScale( width / aWidth, height / aHeight );

	result.translate( aMin, aMMin );

	return result;
}


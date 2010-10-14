#include "GUI/widgets/ScaleVisualizer.h"

ScaleVisualizer::paintEvent( QPaintEvent * event )
{

}

static QTransform
ScaleVisualizer::PrepareCanvasTransform( double aMin, double aMax, double aMMin, double aMMax, int aWidth, int aHeight )
{
	double width = aMax - aMin;
	double height = aMMax - aMMin;
	QTransform result = QTransform::fromScale( width / aWidth, height / aHeight );
}


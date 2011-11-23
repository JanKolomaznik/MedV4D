#include "GUI/widgets/ScaleVisualizer.h"
#include "MedV4D/Common/Common.h"


void
ScaleVisualizer::UpdateSettings()
{
	UpdateTransform();
}

void
ScaleVisualizer::UpdateTransform()
{
	mCanvasTransform = PrepareCanvasTransform( mMin, mMax, mMMin, mMMax, width(), height(), mBorderWidth );
	mInversionTransform = mCanvasTransform.inverted();

	mPixelSize = QPointF( 1,1) * mInversionTransform - QPointF( 0,0) * mInversionTransform;
}

void
ScaleVisualizer::paintEvent( QPaintEvent * event )
{

}

QTransform
ScaleVisualizer::PrepareCanvasTransform( double aMin, double aMax, double aMMin, double aMMax, int aWidth, int aHeight, int aBorderWidth )
{
	double width = aMax - aMin;
	double height = aMMax - aMMin;
	QTransform trans = QTransform::fromTranslate( aBorderWidth, aHeight - aBorderWidth );
	QTransform result = QTransform::fromScale( (aWidth  - 2*aBorderWidth) / (float)width, -1.0 * (aHeight - 2*aBorderWidth) / (float)height );

	//result.translate( aMin/* + aBorderWidth*/, aMMin/* + aBorderWidth*/ );

	return result * trans;
}


#ifndef SCALE_VISUALIZER_H
#define SCALE_VISUALIZER_H

#include <QtWidgets>


class ScaleVisualizer: public QWidget
{
	Q_OBJECT;
public:
	ScaleVisualizer( QWidget *parent = NULL ): QWidget( parent ), mMouseDown( false ), mBorderWidth( 0 )
		{ /*empty*/ }

public slots:
	void
	SetValueInterval( double aMin, double aMax )
	{
		mMin = aMin;
		mMax = aMax;
		UpdateSettings();
	}

	void
	SetMappedValueInterval( double aMin, double aMax )
	{
		mMMin = aMin;
		mMMax = aMax;
		UpdateSettings();
	}

	void
	SetBorderWidth( int aBorderWidth )
	{
		mBorderWidth = aBorderWidth;
		UpdateSettings();
	}
signals:

protected:
	virtual void
	UpdateSettings();
	virtual void
	UpdateTransform();

	static QTransform
	PrepareCanvasTransform( double aMin, double aMax, double aMMin, double aMMax, int aWidth, int aHeight, int aBorderWidth = 0 );

	template< typename TIterator, typename TValueAccessor >
	void
	DrawPolyline(
			QPainter &aPainter,
			TIterator aFirst,
			TIterator aLast,
			double aStep,
			double aStart,
			TValueAccessor aValueAccesor
			);


	void
	paintEvent( QPaintEvent * event );

	QPainter	mPainter;
	QTransform	mCanvasTransform;
	QTransform 	mInversionTransform;
	QPointF		mPixelSize;

	QPointF		mLastPoint;
	QPointF		mCurrentPoint;
	bool		mMouseDown;

	int mBorderWidth;
	double mMin;
	double mMax;
	double mMMin;
	double mMMax;
private:
};

template< typename TIterator, typename TValueAccessor >
void
ScaleVisualizer::DrawPolyline(
		QPainter &aPainter,
		TIterator aFirst,
		TIterator aLast,
		double aStep,
		double aStart,
		TValueAccessor aValueAccesor
		)
{
	if ( aFirst == aLast ) {
		return;
	}

	double p1x, p1y, p2x, p2y;

	p2x = aStart;
	p2y = aValueAccesor( *aFirst );
	++aFirst;

	while( aFirst != aLast ) {
		p1x = p2x;
		p1y = p2y;

		p2x += aStep;
		p2y = aValueAccesor( *aFirst );

		aPainter.drawLine( QLineF( p1x, p1y, p2x, p2y ) );

		++aFirst;
	}

}


#endif /*SCALE_VISUALIZER_H*/

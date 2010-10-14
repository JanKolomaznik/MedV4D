#ifndef SCALE_VISUALIZER_H
#define SCALE_VISUALIZER_H

#include <QtGui>


class ScaleVisualizer: public QWidget
{
	Q_OBJECT;
public:
	ScaleVisualizer( QWidget *parent = NULL ): QWidget( parent )
		{ /*empty*/ }

public slots:
	void
	SetValueInterval( double aMin, double aMax );

	void
	SetMappedValueInterval( double aMin, double aMax );
signals:

protected:
	static QTransform
	PrepareCanvasTransform( double aMin, double aMax, double aMMin, double aMMax, int aWidth, int aHeight );

	void	
	paintEvent( QPaintEvent * event );

private:
	QTransform mCanvasTransform;
	double mMin;
	double mMax;
	double mMMin;
	double mMMax;
};



#endif /*SCALE_VISUALIZER_H*/

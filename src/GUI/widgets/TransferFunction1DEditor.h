#ifndef TRANSFER_FUNCTION_1D_EDITOR_H
#define TRANSFER_FUNCTION_1D_EDITOR_H

#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/widgets/ScaleVisualizer.h"
#include "Imaging/Histogram.h"

namespace M4D
{
namespace GUI
{
	
class TransferFunction1DEditor: public ScaleVisualizer
{
	Q_OBJECT;
public:
	TransferFunction1DEditor( QWidget *aParent = NULL );

	M4D::GUI::TransferFunctionBuffer1D::Ptr
	GetTransferFunctionBuffer() const
	{
		return mTransferFunctionBuffer;
	}

	M4D::Common::TimeStamp
	GetTimeStamp()
	{
		return mEditTimeStamp;
	}
	
	void
	SetBackgroundHistogram( M4D::Imaging::Histogram64::Ptr aHistogram )
	{
		mBackgroundHistogram = aHistogram;
		if ( mBackgroundHistogram ) {
			mHistogramMaximum = HistogramGetMaxCount( *mBackgroundHistogram );
			mHistogramScaling = 1.0f;
		}
		update();
	}
protected:
	
	static void
	RenderHistogram( QPainter &aPainter, M4D::Imaging::Histogram64 &aHistogram )
	{
		float lastX = aHistogram.GetMin()-1;
		float lastY = (float) aHistogram[ lastX ];
		for( int32 i = aHistogram.GetMin(); i <= aHistogram.GetMax(); ++i ) {
			float tmpY = float(aHistogram[i]);
			aPainter.drawLine( lastX, lastY, float(i), tmpY );
			lastX = i;
			lastY = tmpY;
		}
	}

	void
	UpdateSettings();
	
	void
	UpdateTransform();

	void
	RenderBackground();

	void
	RenderForeground();

	void
	paintEvent( QPaintEvent * event );
	void
	mouseMoveEvent ( QMouseEvent * event );
	void
	mousePressEvent ( QMouseEvent * event );
	void
	mouseReleaseEvent ( QMouseEvent * event );
	void
	wheelEvent ( QWheelEvent * event );

	
	void
	FillTransferFunctionValues( float aLeft, float aLeftVal, float aRight, float aRightVal );

	M4D::GUI::TransferFunctionBuffer1D::Ptr mTransferFunctionBuffer;
	size_t	mEditedChannelIdx;

	M4D::Common::TimeStamp	mEditTimeStamp;

	bool			mIsLineEditing;

	M4D::Imaging::Histogram64::Ptr	mBackgroundHistogram;
	int64			mHistogramMaximum;
	float			mHistogramScaling;
	QTransform		mBackgroundTransform;
private:

};

} /*namespace GUI*/
} /*namespace M4D*/

#endif /*TRANSFER_FUNCTION_1D_EDITOR_H*/

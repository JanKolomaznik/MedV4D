
#include "GUI/widgets/TransferFunction1DEditor.h"

namespace M4D
{
namespace GUI
{
	
TransferFunction1DEditor::TransferFunction1DEditor( QWidget *aParent ): ScaleVisualizer( aParent ), mEditedChannelIdx( 0 )
{
	setMinimumSize ( 256, 256 );


	mTransferFunctionBuffer = M4D::GUI::TransferFunctionBuffer1D::Ptr( new M4D::GUI::TransferFunctionBuffer1D( 4096, Vector2f( 0.0f, 4096.0f ) ) );

	M4D::GUI::TransferFunctionBuffer1D::Iterator it;
	float r,g,b,a;
	float step = 1.0f / (4096.0f);
	int i = 0;
	r = g = b = a = 0.0f;

	float level, width;
	level = 1440;
	width = 350;

	for( it = mTransferFunctionBuffer->Begin(); it != mTransferFunctionBuffer->End(); ++it ) {
		*it = RGBAf( r, g, b, a );
		++i;
		float intensity = ClampToInterval<float>(0.0f, 1.0f, 0.5f + 0.5f*((float)i - level) / width); 
		r = intensity;
		g = intensity;
		b = intensity;
		a = intensity;
	}
}

void
TransferFunction1DEditor::paintEvent( QPaintEvent * event )
{
	UpdateTransform();

	mPainter.begin( this );
	mPainter.setTransform( mCanvasTransform );

	mPainter.setWorldMatrixEnabled( false );
	mPainter.setClipRegion ( QRegion( mBorderWidth, mBorderWidth, width() - 2 * mBorderWidth + 1, height() - 2 * mBorderWidth + 1 ) );
	mPainter.setWorldMatrixEnabled( true );

	mPainter.setPen ( QApplication::palette().color( QPalette::Midlight ) );

	QBrush brush( QApplication::palette().color( QPalette::Shadow ) );
	mPainter.setBrush( brush );
	mPainter.drawRect( QRectF( mMin, mMMin, mMax - mMin, mMMax - mMMin ) );

	double aStep = (mTransferFunctionBuffer->GetMappedInterval()[1]- mTransferFunctionBuffer->GetMappedInterval()[0]) / (float)mTransferFunctionBuffer->Size();
	double aStart = mTransferFunctionBuffer->GetMappedInterval()[0];
	mPainter.setPen ( QColor( 255, 0, 0, 255 ) );
	DrawPolyline(
			mPainter,
			mTransferFunctionBuffer->Begin(),
			mTransferFunctionBuffer->End(),
			aStep,
			aStart,
			RGBAf::ValueAccessor< 0 >()
			);
	mPainter.setPen ( QColor( 0, 255, 0, 255 ) );
	DrawPolyline(
			mPainter,
			mTransferFunctionBuffer->Begin(),
			mTransferFunctionBuffer->End(),
			aStep,
			aStart,
			RGBAf::ValueAccessor< 1 >()
			);
	mPainter.setPen ( QColor( 0, 0, 255, 255 ) );
	DrawPolyline(
			mPainter,
			mTransferFunctionBuffer->Begin(),
			mTransferFunctionBuffer->End(),
			aStep,
			aStart,
			RGBAf::ValueAccessor< 2 >()
			);
	mPainter.setPen ( QApplication::palette().color( QPalette::BrightText ) );
	DrawPolyline(
			mPainter,
			mTransferFunctionBuffer->Begin(),
			mTransferFunctionBuffer->End(),
			aStep,
			aStart,
			RGBAf::ValueAccessor< 3 >()
			);


	mPainter.drawEllipse ( mLastPoint, mPixelSize.x() * 5, mPixelSize.y() * 5 );

	mPainter.end();
}

void
TransferFunction1DEditor::FillTransferFunctionValues( float aLeft, float aLeftVal, float aRight, float aRightVal )
{
	int first = mTransferFunctionBuffer->GetNearestIndex( aLeft );
	int last = mTransferFunctionBuffer->GetNearestIndex( aRight );

	if ( first == -1 && last == -1 ) {
		return;
	}
	if ( first == (int)mTransferFunctionBuffer->Size() && last == (int)mTransferFunctionBuffer->Size() ) {
		return;
	}
	if ( first == -1 ) {
		first = 0;
	}
	if ( last == (int)mTransferFunctionBuffer->Size() ) {
		--last;
	}
	//TODO fix for right interval
	float step = (aRightVal - aLeftVal) / (last - first);
	
	for ( ; first <= last; ++first, aLeftVal += step ) {
		(*mTransferFunctionBuffer)[first][mEditedChannelIdx] = aLeftVal;
	}
}

void
TransferFunction1DEditor::UpdateSettings()
{
	ScaleVisualizer::UpdateSettings();

	mTransferFunctionBuffer->SetMappedInterval( Vector2f( mMin, mMax ) );
}

void
TransferFunction1DEditor::mouseMoveEvent ( QMouseEvent * event )
{
	if( ! mMouseDown ) {
		return;
	}
	mIsLineEditing = event->modifiers() & Qt::ControlModifier;
	if ( !mIsLineEditing ) {
		mLastPoint = mCurrentPoint;
	}
	mCurrentPoint = event->posF()*mInversionTransform;

	float left;
	float right;
	float leftVal;
	float rightVal;

	if ( mLastPoint.x() < mCurrentPoint.x() ) {
		left = mLastPoint.x() - mPixelSize.x() * 0.5f;
		right = mCurrentPoint.x() + mPixelSize.x() * 0.5f;
		leftVal = ClampToInterval( 0.0f, 1.0f, (float)mLastPoint.y() );
		rightVal = ClampToInterval( 0.0f, 1.0f, (float)mCurrentPoint.y() );
	} else {
		left = mCurrentPoint.x() - mPixelSize.x() * 0.5f;
		right = mLastPoint.x() + mPixelSize.x() * 0.5f;
		leftVal = ClampToInterval( 0.0f, 1.0f, (float)mCurrentPoint.y() );
		rightVal = ClampToInterval( 0.0f, 1.0f, (float)mLastPoint.y() );
	}
	
	FillTransferFunctionValues( left, leftVal, right, rightVal );

	++mEditTimeStamp;


	update();
}

void
TransferFunction1DEditor::mousePressEvent ( QMouseEvent * event )
{
	if ( event->button() == Qt::RightButton ) {
		mEditedChannelIdx = (mEditedChannelIdx + 1) % 4;
		update();
		return;
	}
	mLastPoint = mCurrentPoint = event->posF()*mInversionTransform;

	mMouseDown = true;

	mIsLineEditing = event->modifiers() & Qt::ControlModifier;

	update();
}

void
TransferFunction1DEditor::mouseReleaseEvent ( QMouseEvent * event )
{
	mMouseDown = false;
	update();
}

} /*namespace GUI*/
} /*namespace M4D*/

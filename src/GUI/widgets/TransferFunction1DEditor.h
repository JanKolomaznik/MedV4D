#ifndef TRANSFER_FUNCTION_1D_EDITOR_H
#define TRANSFER_FUNCTION_1D_EDITOR_H

#include "GUI/utils/TransferFunctionBuffer.h"
#include "GUI/widgets/ScaleVisualizer.h"

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

protected:
	
	void
	UpdateSettings();


	void
	paintEvent( QPaintEvent * event );
	void
	mouseMoveEvent ( QMouseEvent * event );
	void
	mousePressEvent ( QMouseEvent * event );
	void
	mouseReleaseEvent ( QMouseEvent * event );

	
	void
	FillTransferFunctionValues( float aLeft, float aLeftVal, float aRight, float aRightVal );

	M4D::GUI::TransferFunctionBuffer1D::Ptr mTransferFunctionBuffer;
	size_t	mEditedChannelIdx;

	M4D::Common::TimeStamp	mEditTimeStamp;

	bool			mIsLineEditing;
private:

};

} /*namespace GUI*/
} /*namespace M4D*/

#endif /*TRANSFER_FUNCTION_1D_EDITOR_H*/

#ifndef TRANSFER_FUNCTION
#define TRANSFER_FUNCTION

namespace M4D {
namespace GUI {


class ATransferFunction1D
{
public:
	virtual 
	~ATransferFunction()

	virtual RGBAf
	GetMappedRGBA( float32 )const = 0;
	
	virtual HSVAf
	GetMappedHSVA( float32 )const = 0;

protected:
	ATransferFunction( float32 aMinimum, float32 aMaximum );

	/*size_t	mStepCount;
	RGBAf	*mBuffer;*/
};

} //namespace M4D
} //namespace GUI

#endif /*TRANSFER_FUNCTION*/

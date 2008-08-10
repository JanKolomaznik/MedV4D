#ifndef _IMAGE_CONVERTOR_H
#define _IMAGE_CONVERTOR_H

#include "Common.h"
#include "Imaging/AbstractFilter.h"


namespace M4D
{

namespace Imaging
{

class DefaultConvertor
{
public:
	template< typename InputType, typename OutputType >
	void
	operator()( const InputType &input, OutputType &output )
		{
			output = (OutputType)input;
		}
};

template< typename OutputImageType, typename Convertor = DefaultConvertor >
class ImageConvertor;

template< typename OutputElementType, typename Convertor >
class ImageConvertor< Image< OutputElementType, 2 > >
{
	//TODO
};

template< typename InputElementType >
class ImageConvertor< Image< InputElementType, 3 > >
	: public AbstractPipeFilter
{
public:
	typedef AbstractPipeFilter 	PredecessorType;

	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};

	ImageConvertor( Properties  * prop );
	ImageConvertor();
protected:


private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

	
} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/filters/ImageConvertor.tcc"

#endif /*_IMAGE_CONVERTOR_H*/

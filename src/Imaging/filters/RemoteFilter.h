#ifndef _REMOTE_FILTER_H
#define _REMOTE_FILTER_H

#include "Common.h"


namespace M4D
{

namespace Imaging
{


struct RemoteFilterOptions : public AbstractFilterSettings
{

};



template< typename InputImageType, typename OutputImageType >
class RemoteFilter 
	: public ImageFilterWholeAtOnce< InputImageType, OutputImageType >
{
public:
	typedef RemoteFilterOptions	Settings;

	RemoteFilter();
	~RemoteFilter();
protected:
	typedef typename  Imaging::ImageFilterWholeAtOnce< InputImageType, OutputImageType > PredecessorType;

	bool
	ProcessImage(
			const InputImageType 	&in,
			OutputImageType		&out
		    );

	void
	PrepareOutputDatasets();

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/filters/RemoteFilter.tcc"

#endif /*_REMOTE_FILTER_H*/

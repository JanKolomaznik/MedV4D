#ifndef FILTERING_H
#define FILTERING_H

#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageConvertor.h"

template< typename ImageType >
M4D::Imaging::PipelineContainer *
PreparePipeline( 
		M4D::Imaging::AbstractPipeFilter 		&filter, 
		M4D::Imaging::MessageReceiverInterface::Ptr 	hook,
		M4D::Imaging::AbstractImageConnectionInterface 	*inConnection,
		M4D::Imaging::AbstractImageConnectionInterface 	*outConnection
		)
{
	M4D::Imaging::PipelineContainer *container = new PipelineContainer();
	M4D::Imaging::ImageConvertor< ImageType > *convertor = new M4D::Imaging::ImageConvertor< ImageType >();

	try {
		container.AddFilter( convertor );
		container.AddFilter( &filter );

		filter.SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );

		container.MakeConnection( *convertor, 0, filter, 0 );
		
		inConnection = &( container.MakeInputConnection( *convertor, 0, false ) );

		outConnection = &( container.MakeOutputConnection( filter, 0, false ) );

		outConnection->SetMessageHook( hook );
	} 
	catch( ... ) {
		delete container;
		delete convertor;
		throw;
	}

	return container;
}

#endif /*FILTERING_H*/

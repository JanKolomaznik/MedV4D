#ifndef FILTERING_H
#define FILTERING_H

#include "Imaging/AbstractFilter.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/filters/ImageConvertor.h"

class FinishHook: public M4D::Imaging::MessageReceiverInterface
{
public:
	FinishHook() : _finished( false ) {}

	void
	ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr 			msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle 	sendStyle, 
		M4D::Imaging::FlowDirection				direction
		)
		{
			_finished = msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED;	
		}

	bool
	Finished()
		{ return _finished; }
private:
	bool _finished;
};


template< typename ImageType >
M4D::Imaging::PipelineContainer *
PreparePipeline( 
		M4D::Imaging::AbstractPipeFilter 		&filter, 
		M4D::Imaging::MessageReceiverInterface::Ptr 	hook,
		M4D::Imaging::AbstractImageConnectionInterface 	*inConnection,
		M4D::Imaging::AbstractImageConnectionInterface 	*outConnection
		)
{
	M4D::Imaging::PipelineContainer *container = new M4D::Imaging::PipelineContainer();
	M4D::Imaging::ImageConvertor< ImageType > *convertor = new M4D::Imaging::ImageConvertor< ImageType >();

	try {
		container->AddFilter( convertor );
		container->AddFilter( &filter );

		filter.SetUpdateInvocationStyle( M4D::Imaging::AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );

		container->MakeConnection( *convertor, 0, filter, 0 );
		
		inConnection = dynamic_cast<M4D::Imaging::AbstractImageConnectionInterface*>( 
				&( container->MakeInputConnection( *convertor, 0, false ) ) 
				);

		outConnection = dynamic_cast<M4D::Imaging::AbstractImageConnectionInterface*>( 
				&( container->MakeOutputConnection( filter, 0, false ) ) 
				);

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

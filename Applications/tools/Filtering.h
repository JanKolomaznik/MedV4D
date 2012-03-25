#ifndef FILTERING_H
#define FILTERING_H

#include "MedV4D/Imaging/AFilter.h"
#include "MedV4D/Imaging/PipelineContainer.h"
#include "MedV4D/Imaging/filters/ImageConvertor.h"

class FinishHook: public M4D::Imaging::MessageReceiverInterface
{
public:
	FinishHook() : _finished( false ), _OK( true ) {}

	void
	ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr 			msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle 	sendStyle, 
		M4D::Imaging::FlowDirection				direction
		)
		{
			if( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) {
				_finished = true;
				_OK = true;
				return;
			}
			if( msg->msgID == M4D::Imaging::PMI_FILTER_CANCELED ) {
				_finished = true;
				_OK = false;
				return;
			}					
		}

	bool
	Finished()
		{ return _finished; }
	bool
	OK()
		{ return _OK; }
private:
	volatile bool _finished;
	volatile bool _OK;
};


template< typename ImageType >
M4D::Imaging::PipelineContainer *
PreparePipeline( 
		M4D::Imaging::APipeFilter 		&filter, 
		M4D::Imaging::MessageReceiverInterface::Ptr 	hook,
		M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > 	*&inConnection,
		M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > 	*&outConnection
		)
{
	M4D::Imaging::PipelineContainer *container = new M4D::Imaging::PipelineContainer();
	M4D::Imaging::ImageConvertor< ImageType > *convertor = new M4D::Imaging::ImageConvertor< ImageType >();

	try {
		container->AddFilter( convertor );
		container->AddFilter( &filter );

		//filter.SetUpdateInvocationStyle( M4D::Imaging::APipeFilter::UIS_ON_CHANGE_BEGIN );
		filter.SetUpdateInvocationStyle( M4D::Imaging::APipeFilter::UIS_ON_UPDATE_FINISHED );

		container->MakeConnection( *convertor, 0, filter, 0 );
		
		inConnection = dynamic_cast<M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >* >( 
				&( container->MakeInputConnection( *convertor, 0, false ) ) 
				);

		outConnection = dynamic_cast<M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >* >( 
				&( container->MakeOutputConnection( filter, 0, true ) ) 
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

M4D::Imaging::PipelineContainer *
PrepareSimplePipeline( 
		M4D::Imaging::APipeFilter 		&filter, 
		M4D::Imaging::MessageReceiverInterface::Ptr 	hook,
		M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > 	*&inConnection,
		M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > 	*&outConnection
		)
{
	M4D::Imaging::PipelineContainer *container = new M4D::Imaging::PipelineContainer();
	try {
		container->AddFilter( &filter );

		//filter.SetUpdateInvocationStyle( M4D::Imaging::APipeFilter::UIS_ON_CHANGE_BEGIN );
		//filter.SetUpdateInvocationStyle( M4D::Imaging::APipeFilter::UIS_ON_UPDATE_FINISHED );

		inConnection = dynamic_cast< M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >* >( 
				&( container->MakeInputConnection( filter, 0, false ) ) 
				);

		outConnection = dynamic_cast< M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >* >( 
				&( container->MakeOutputConnection( filter, 0, true ) ) 
				);

		outConnection->SetMessageHook( hook );
	} 
	catch( ... ) {
		delete container;
		throw;
	}

	return container;
}

#endif /*FILTERING_H*/

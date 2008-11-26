#include "Common.h"
#include "Filtering.h"


using namespace M4D;
using namespace M4D::Imaging;

typedef Image< int16, 3 > ImageType;

class FinishHook: public M4D::Imaging::MessageReceiverInterface
{
	FinishHook() : _finished( false );

	void
	ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle, 
		FlowDirection				direction
		)
		{
			_finished = msg->msgID == PMI_FILTER_UPDATED;	
		}

	bool
	Finished()
		{ return _finished; }
private:
	bool _finished;
};

int
main( int argc, char **argv )
{

	std::string inFilename = argv[1];
	std::string outFilename = argv[2];


	std::cout << "Loading file...";
	M4D::Imaging::AbstractImage::AImagePtr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Done\n";


	std::cout << "Initializing...\n";
	M4D::Imaging::PipelineContainer *container = NULL;
	FinishHook  *hook = new FinishHook;
	M4D::Imaging::AbstractImageConnectionInterface *inConnection = NULL;
	M4D::Imaging::AbstractImageConnectionInterface *outConnection = NULL;
	/*---------------------------------------------------------------------*/


	/*---------------------------------------------------------------------*/
	container = PreparePipeline<ImageType>( &filter, M4D::Imaging::MessageReceiverInterface::Ptr( hook ), inConnection, *outConnection );
	std::cout << "Done\n";

	std::cout << "Computing...\n";
	container->ExecuteFirstFilter();

	while( ! FinishHook->Finished() ){ /*empty*/ }

	std::cout << "Done\n";

	std::cout << "Saving file...";
	M4D::Imaging::ImageFactory::DumpImage( outFileName, outConnection->GetAbstractImageReadOnly() );
	std::cout << "Done\n";

	delete container;

	return 0;
}
#ifndef DATASETFACTORY_H_
#define DATASETFACTORY_H_

#include "ADataset.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/GeometryDatasetFactory.h"


namespace M4D
{
namespace Imaging
{

/**
 * Factory class that creates data sets of all kind. 
 */
class DatasetFactory : public ImageFactory, public GeometryDatasetFactory
{
public:
	/**
	 * Creates data set based on atributes that reads from stream
	 */
	static ADataset::Ptr 
	DeserializeDataset(M4D::IO::InStream &stream);

	static void
	DeserializeDataset(M4D::IO::InStream &stream, ADataset &dataset);

	static void 
	SerializeDataset(M4D::IO::OutStream &stream, const ADataset &dataset);
	
	
private:	// helpers
	/*static ADataset::Ptr 
	CreateImage(M4D::IO::InStream &stream);*/
};

}
}
#endif /*DATASETFACTORY_H_*/

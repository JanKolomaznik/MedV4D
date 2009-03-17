#ifndef DATASETFACTORY_H_
#define DATASETFACTORY_H_

#include "AbstractDataSet.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/GeometryDataSetFactory.h"


namespace M4D
{
namespace Imaging
{

/**
 * Factory class that creates data sets of all kind. 
 */
class DataSetFactory : public ImageFactory, public GeometryDataSetFactory
{
public:
	/**
	 * Creates data set based on atributes that reads from stream
	 */
	static AbstractDataSet::Ptr 
	CreateDataSet(M4D::IO::InStream &stream);
	
private:	// helpers
	static AbstractDataSet::Ptr 
	CreateImage(M4D::IO::InStream &stream);
};

}
}
#endif /*DATASETFACTORY_H_*/

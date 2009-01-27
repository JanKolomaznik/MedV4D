#ifndef DATASETFACTORY_H_
#define DATASETFACTORY_H_

#include "iAccessStream.h"
#include "AbstractDataSet.h"
#include "Imaging/ImageFactory.h"

namespace M4D
{
namespace Imaging
{

/**
 * Factory class that creates data sets of all kind. 
 */
class DataSetFactoryA : public ImageFactory
{
public:
	/**
	 * Creates data set based on atributes that reads from stream
	 */
	static AbstractDataSet::ADataSetPtr CreateDataSet(iAccessStream &stream);
	
private:	// helpers
	static AbstractDataSet::ADataSetPtr CreateImage(iAccessStream &stream);
};

}
}
#endif /*DATASETFACTORY_H_*/

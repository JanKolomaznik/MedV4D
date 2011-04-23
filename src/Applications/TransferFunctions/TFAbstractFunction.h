#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <QtCore/QString>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>

#include <TFCommon.h>
#include <TFColor.h>
#include <TFFunctionInterface.h>
#include <TFColorVector.h>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFAbstractFunction: public TFFunctionInterface{

public:

	typedef typename boost::shared_ptr<TFAbstractFunction<dim>> Ptr;

	virtual ~TFAbstractFunction(){}

	TFFunctionInterface::Ptr clone() = 0;

	TF::Color& color(const TF::Coordinates& coords){

		return colorMap_->value(coords);
	}

	virtual TF::Color getRGBfColor(const TF::Coordinates& coords) = 0;
	virtual void setRGBfColor(const TF::Coordinates& coords, const TF::Color& value) = 0;
	
	TF::Size getDimension(){

		return dim;
	}
	
	TF::Size getDomain(const TF::Size dimension){

		return colorMap_->size(dimension);
	}
	
	void resize(const std::vector<TF::Size>& dataStructure){

		colorMap_->recalculate(dataStructure);
	}
	 
	void save(TF::XmlWriterInterface* writer){

		saveSettings_(writer);

		writer->writeStartElement("Function");

			writer->writeAttribute("Dimensions", TF::convert<TF::Size, std::string>(dim));
				
			for(TF::Size i = 0; i < dim; ++i)
			{
				writer->writeAttribute("Dimension" + i,  TF::convert<TF::Size, std::string>(colorMap_->size(i)));
			}
			TF::Coordinates coords(dim);
			saveDimensions_(writer, coords);

		writer->writeEndElement();
	}

	bool load(TF::XmlReaderInterface* reader, bool& sideError){

		#ifndef TF_NDEBUG
			std::cout << "Loading function..." << std::endl;
		#endif
			
		sideError = !loadSettings_(reader);

		if(reader->readElement("Function"))
		{		
			TF::Size dimControl = TF::convert<std::string, TF::Size>(
				reader->readAttribute("Dimensions"));
			if(dimControl != dim) return false;

			std::vector<TF::Size> dataStructure;
			for(TF::Size i = 0; i < dim; ++i)
			{
				dataStructure.push_back(TF::convert<std::string, TF::Size>(
					reader->readAttribute("Dimension" + i)));
			}
			resize(dataStructure);

			TF::Coordinates coords(dim);
			return loadDimensions_(reader, coords);
		}
		return false;
	}

protected:

	typename TF::ColorVector<dim>::Ptr colorMap_;

	TFAbstractFunction(const std::vector<TF::Size>& domains):
		colorMap_(new TF::ColorVector<dim>(domains)){
	}

	TFAbstractFunction(){}

	virtual void saveSettings_(TF::XmlWriterInterface* writer){}
	virtual bool loadSettings_(TF::XmlReaderInterface* reader){ return true; }

	void saveDimensions_(TF::XmlWriterInterface* writer,
			TF::Coordinates& coords,
			const TF::Size currentDim = 1){

		if(currentDim == dim)
		{
			std::string strCoords = "[";
			for(TF::Size i = 0; i < dim - 1; ++i) strCoords += coords[i] + ",";
			strCoords += coords[dim - 1] + "]";

			writer->writeStartElement("Color");

				writer->writeAttribute("TF::Coordinates", strCoords);

				writer->writeAttribute("Component1",
					TF::convert<float, std::string>(colorMap_->value(coords).component1));
				writer->writeAttribute("Component2",
					TF::convert<float, std::string>(colorMap_->value(coords).component2));
				writer->writeAttribute("Component3",
					TF::convert<float, std::string>(colorMap_->value(coords).component3));
				writer->writeAttribute("Alpha",
					TF::convert<float, std::string>(colorMap_->value(coords).alpha));
		
			writer->writeEndElement();

			return;
		}

		for(TF::Size i = 0; i < colorMap_->size(currentDim); ++i)
		{
			saveDimensions_(writer, coords, currentDim + 1);

			++coords[currentDim - 1];
			for(TF::Size j = currentDim; j < dim; ++j)
			{
				coords[j] = 0;
			}
		}
	}

	bool loadDimensions_(TF::XmlReaderInterface* reader,
			TF::Coordinates& coords,
			const TF::Size currentDim = 1){

		if(currentDim == dim)
		{
			if(reader->readElement("Color"))
			{		
				colorMap_->value(coords).component1 = TF::convert<std::string, float>(
					reader->readAttribute("Component1"));
				colorMap_->value(coords).component2 = TF::convert<std::string, float>(
					reader->readAttribute("Component2"));
				colorMap_->value(coords).component3 = TF::convert<std::string, float>(
					reader->readAttribute("Component3"));
				colorMap_->value(coords).alpha = TF::convert<std::string, float>(
					reader->readAttribute("Alpha"));
				return true;
			}
			return false;
		}

		bool ok = true;
		bool indexOK;
		for(TF::Size i = 0; i < colorMap_->size(currentDim); ++i)
		{
			indexOK = loadDimensions_(reader, coords, currentDim + 1);
			ok = ok && indexOK;

			++coords[currentDim - 1];
			for(TF::Size j = currentDim; j < dim; ++j)
			{
				coords[j] = 0;
			}
		}
		return ok;
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION
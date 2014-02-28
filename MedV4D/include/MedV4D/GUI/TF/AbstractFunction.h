#ifndef TF_ABSTRACUNCTION
#define TF_ABSTRACUNCTION

#include <QtCore/QString>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/Color.h"
#include "MedV4D/GUI/TF/FunctionInterface.h"
#include "MedV4D/GUI/TF/ColorVector.h"

namespace M4D {
namespace GUI {

template<TF::Size dim>
class AbstractFunction: public FunctionInterface{

public:

	typedef boost::shared_ptr< AbstractFunction<dim> > Ptr;

	virtual ~AbstractFunction(){}

	FunctionInterface::Ptr clone() = 0;

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

		writer->writeStartElement("Function");

			writer->writeAttribute("Dimensions", TF::convert<TF::Size, std::string>(dim));

			for(TF::Size i = 1; i <= dim; ++i)
			{
				writer->writeAttribute("Dimension" + TF::convert<TF::Size, std::string>(i),
					TF::convert<TF::Size, std::string>(colorMap_->size(i)));
			}
			TF::Coordinates coords(dim);
			saveDimensions_(writer, coords);

		writer->writeEndElement();

		saveSettings_(writer);
	}

	bool load(TF::XmlReaderInterface* reader){

		#ifndef TF_NDEBUG
			std::cout << "Loading function..." << std::endl;
		#endif

		bool ok = false;
		if(reader->readElement("Function"))
		{
			TF::Size dimControl = TF::convert<std::string, TF::Size>(
				reader->readAttribute("Dimensions"));
			if(dimControl != dim) return false;

			std::vector<TF::Size> dataStructure;
			for(TF::Size i = 1; i <= dim; ++i)
			{
				dataStructure.push_back(TF::convert<std::string, TF::Size>(
					reader->readAttribute("Dimension" + TF::convert<TF::Size, std::string>(i))));
			}
			resize(dataStructure);

			TF::Coordinates coords(dim);
			ok = loadDimensions_(reader, coords);

			bool settingsLoaded = loadSettings_(reader);
			ok = ok && settingsLoaded;
		}

		return ok;
	}

protected:

	typename TF::ColorVector<dim>::Ptr colorMap_;

	AbstractFunction(const std::vector<TF::Size>& domains):
		colorMap_(new TF::ColorVector<dim>(domains)){
	}

	AbstractFunction(){}

	virtual void saveSettings_(TF::XmlWriterInterface* writer){}
	virtual bool loadSettings_(TF::XmlReaderInterface* reader){ return true; }

	void saveDimensions_(TF::XmlWriterInterface* writer,
			TF::Coordinates& coords,
			const TF::Size currentDim = 1){

		if(currentDim == dim)
		{
			for(TF::Size i = 0; i < colorMap_->size(currentDim); ++i)
			{
				std::string strCoords = "[";
				for(TF::Size j = 0; j < dim - 1; ++j) strCoords += TF::convert<int, std::string>(coords[j]) + ",";
				strCoords += TF::convert<int, std::string>(coords[dim - 1]) + "]";

				writer->writeStartElement("Color");

					writer->writeAttribute("Coordinates", strCoords);

					writer->writeAttribute("Component1",
						TF::convert<float, std::string>(colorMap_->value(coords).component1));
					writer->writeAttribute("Component2",
						TF::convert<float, std::string>(colorMap_->value(coords).component2));
					writer->writeAttribute("Component3",
						TF::convert<float, std::string>(colorMap_->value(coords).component3));
					writer->writeAttribute("Alpha",
						TF::convert<float, std::string>(colorMap_->value(coords).alpha));

				writer->writeEndElement();

				++coords[currentDim - 1];
			}

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

		bool ok = true;

		if(currentDim == dim)
		{
			for(TF::Size i = 0; i < colorMap_->size(currentDim); ++i)
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
				}
				else
				{
					ok = false;
				}

				++coords[currentDim - 1];
			}

			return ok;
		}

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

#endif //TF_ABSTRACUNCTION

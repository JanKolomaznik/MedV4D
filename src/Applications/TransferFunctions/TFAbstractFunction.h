#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <QtCore/QString>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>

#include <TFCommon.h>
#include <TFColor.h>

namespace M4D {
namespace GUI {

class TFFunctionConstAccesor;

class TFFunctionInterface{

public:

	typedef boost::shared_ptr<TFFunctionInterface> Ptr;
	typedef TFFunctionConstAccesor Const;

	virtual ~TFFunctionInterface(){}

	static const TF::Size defaultDomain = 4095;	//TODO ?

	virtual Ptr clone() = 0;
	
	virtual TF::Color& color(const TF::Size dimension, const TF::Size index) = 0;

	virtual TF::Color getRGBfColor(const TF::Size dimension, const TF::Size index) = 0;
	virtual void setRGBfColor(const TF::Size dimension, const TF::Size index, const TF::Color& value) = 0;

	virtual TF::Size getDomain(const TF::Size dimension) = 0;
	virtual TF::Size getDimension() = 0;

	virtual void clear(const TF::Size dimension) = 0;
	virtual void resize(const std::vector<TF::Size>& dataStructure) = 0;

	virtual void save(TF::XmlWriterInterface* writer) = 0;
	virtual bool load(TF::XmlReaderInterface* reader, bool& sideError) = 0;

protected:

	TFFunctionInterface(){}
};

class TFFunctionConstAccesor{

	friend class TFFunctionInterface;

public:

	const TF::Color nullColor;

	TFFunctionConstAccesor():
		nullColor(-1,-1,-1,-1),
		null_(true){
	}

	TFFunctionConstAccesor(TFFunctionInterface::Ptr function):
		function_(function),
		nullColor(-1,-1,-1,-1),
		null_(false){
	}
		
	TFFunctionConstAccesor(TFFunctionInterface* function):
		function_(TFFunctionInterface::Ptr(function)),
		nullColor(-1,-1,-1,-1),
		null_(false){
	}
	~TFFunctionConstAccesor(){}

	void operator=(const TFFunctionConstAccesor &constPtr){

		function_ = constPtr.function_;
		null_ = constPtr.null_;
	}

	bool operator==(const int &null){

		return null_;
	}
	
	const TF::Color& color(const TF::Size dimension, const TF::Size index){

		if(null_) return nullColor;
		return function_->color(dimension, index);
	}

	TF::Color getRGBfColor(const TF::Size dimension, const TF::Size index){

		if(null_) return nullColor;
		return function_->getRGBfColor(dimension, index);
	}

	TF::Size getDomain(const TF::Size dimension){

		if(null_) return 0;
		return function_->getDomain(dimension);
	}

	TF::Size getDimension(){

		if(null_) return 0;
		return function_->getDimension();
	}

private:

	bool null_;
	TFFunctionInterface::Ptr function_;
};

template<TF::Size dim>
class TFAbstractFunction: public TFFunctionInterface{

public:

	typedef typename boost::shared_ptr<TFAbstractFunction<dim>> Ptr;

	TFAbstractFunction(const std::vector<TF::Size>& domains){

		for(TF::Size i = 0; i < dim; ++i)
		{
			if(i < domains.size()) colorMap_[i] = TF::Color::MapPtr(new TF::Color::Map(domains[i]));
			else colorMap_[i] = TF::Color::MapPtr(new TF::Color::Map(TFFunctionInterface::defaultDomain));
			clear(i+1);
		}
	}

	virtual ~TFAbstractFunction(){}

	TFFunctionInterface::Ptr clone(){

		return TFFunctionInterface::Ptr(new TFAbstractFunction<dim>(*this));
	}

	TF::Color& color(const TF::Size dimension, const TF::Size index){

		TF::Size innerDimension = dimension - 1;

		return (*colorMap_[innerDimension])[index];
	}

	virtual TF::Color getRGBfColor(const TF::Size dimension, const TF::Size index){

		return color(dimension, index); 
	}

	virtual void setRGBfColor(const TF::Size dimension, const TF::Size index, const TF::Color& value){

		color(dimension, index) = value; 
	}
	
	TF::Size getDimension(){

		return dim;
	}
	
	TF::Size getDomain(const TF::Size dimension){

		TF::Size innerDimension = dimension - 1;

		return colorMap_[innerDimension]->size();
	}

	void clear(const TF::Size dimension){

		TF::Size innerDimension = dimension - 1;

		TF::Color::Map::iterator begin = colorMap_[innerDimension]->begin();
		TF::Color::Map::iterator end = colorMap_[innerDimension]->end();
		for(TF::Color::Map::iterator it = begin; it!=end; ++it)
		{
			*it = TF::Color();
 		}
	}
	
	void resize(const std::vector<TF::Size>& dataStructure){
		
		for(TF::Size i = 0; i < dataStructure.size(); ++i)
		{
			if(dataStructure[i] == colorMap_[i]->size()) return;

			TF::Color::MapPtr resized(new TF::Color::Map(dataStructure[i]));
			resize_(colorMap_[i], resized);

			colorMap_[i] = resized;
		}
	}
	 
	void save(TF::XmlWriterInterface* writer){

		saveSettings_(writer);

		writer->writeStartElement("Function");

			writer->writeAttribute("Dimensions", TF::convert<TF::Size, std::string>(dim));
				
			for(TF::Size i = 0; i < dim; ++i)
			{
				writer->writeStartElement("Dimension");
				writer->writeAttribute("Number", TF::convert<TF::Size, std::string>(i));
				writer->writeAttribute("Domain", TF::convert<TF::Size, std::string>(colorMap_[i]->size()));

				for(TF::Size j = 0; j < colorMap_[i]->size(); ++j)
				{
						writer->writeStartElement("Color");

							writer->writeAttribute("Component1",
								TF::convert<float, std::string>((*colorMap_[i])[j].component1));
							writer->writeAttribute("Component2",
								TF::convert<float, std::string>((*colorMap_[i])[j].component2));
							writer->writeAttribute("Component3",
								TF::convert<float, std::string>((*colorMap_[i])[j].component3));
							writer->writeAttribute("Alpha",
								TF::convert<float, std::string>((*colorMap_[i])[j].alpha));
					
						writer->writeEndElement();
				}
				writer->writeEndElement();
			}

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

			TF::Size dimension;
			TF::Size domain;
			for(TF::Size i = 0; i < dim; ++i)
			{
				dimension = TF::convert<std::string, TF::Size>(
					reader->readAttribute("Number"));
				if(dimension != i+1) return false;
				domain = TF::convert<std::string, TF::Size>(
					reader->readAttribute("Domain"));
				if(domain == 0) return false;

				TF::Color::MapPtr loaded(new TF::Color::Map(domain));			
				TF::Size j = 0;
				for(; j < domain; ++j)
				{
					if(!loadColor_(reader, loaded, j)) break;
				}

				if(j != domain) return false;
					
				resize_(loaded, colorMap_[i]);
			}
		}
		return true;
	}

protected:

	TF::Color::MapPtr colorMap_[dim];

	void resize_(const TF::Color::MapPtr old, TF::Color::MapPtr resized){

		int inputSize = old->size();
		int outputSize = resized->size();

		float correction = outputSize/(float)inputSize;

		if(correction >= 1)
		{
			int ratio = (int)(correction);	//how many old values are used for computing 1 resized values
			correction -= ratio;
			float corrStep = correction;

			int outputIndexer = 0;
			for(int inputIndexer = 0; inputIndexer < inputSize; ++inputIndexer)
			{
				TF::Size valueCount = ratio + (int)correction;
				for(TF::Size i = 0; i < valueCount; ++i)
				{
					//tfAssert(outputIndexer < outputSize);
					if(inputIndexer >= inputSize) break;

					(*resized)[outputIndexer] = (*old)[inputIndexer];

					++outputIndexer;
				}
				correction -= (int)correction;
				correction += corrStep;
			}
		}
		else
		{
			correction = inputSize/(float)outputSize;
			int ratio =  (int)(correction);	//how many old values are used for computing 1 resized values
			correction -= ratio;
			float corrStep = correction;

			int inputIndexer = 0;
			for(int outputIndexer = 0; outputIndexer < outputSize; ++outputIndexer)
			{
				TF::Color computedValue;
				TF::Size valueCount = ratio + (int)correction;
				for(TF::Size i = 0; i < valueCount; ++i)
				{
					//tfAssert(inputIndexer < inputSize);
					if(inputIndexer >= inputSize)
					{
						valueCount = i;
						break;
					}

					computedValue += (*old)[inputIndexer];

					++inputIndexer;
				}
				correction -= (int)correction;
				correction += corrStep;

				if(valueCount == 0) break;
				(*resized)[outputIndexer] = computedValue/valueCount;
			}
		}
	}

	virtual void saveSettings_(TF::XmlWriterInterface* writer){}
	virtual bool loadSettings_(TF::XmlReaderInterface* reader){ return true; }

	bool loadColor_(TF::XmlReaderInterface* reader, TF::Color::MapPtr loaded, TF::Size index){

		bool ok = false;
		if(reader->readElement("Color"))
		{		
			(*loaded)[index].component1 = TF::convert<std::string, float>(
				reader->readAttribute("Component1"));
			(*loaded)[index].component2 = TF::convert<std::string, float>(
				reader->readAttribute("Component2"));
			(*loaded)[index].component3 = TF::convert<std::string, float>(
				reader->readAttribute("Component3"));
			(*loaded)[index].alpha = TF::convert<std::string, float>(
				reader->readAttribute("Alpha"));
			ok = true;
		}
		return ok;
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION
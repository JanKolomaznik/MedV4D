#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <TFXmlReader.h>
#include <TFXmlWriter.h>
#include <QtCore/QString>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>

#include <TFCommon.h>
#include <TFColor.h>

namespace M4D {
namespace GUI {

class TFApplyFunctionInterface{

public:

	typedef boost::shared_ptr<TFApplyFunctionInterface> Ptr;

	static const TF::Size defaultDomain = 4095;	//TODO ?
	
	virtual TF::Color getMappedRGBfColor(const TF::Size value, const TF::Size dimension) = 0;
	virtual TF::Size getDomain() const = 0;
	virtual TF::Size getDimension() const = 0;

protected:

	TFApplyFunctionInterface(){}
	virtual ~TFApplyFunctionInterface(){}
};

template<TF::Size dim>
class TFAbstractFunction: public TFApplyFunctionInterface{

public:

	typedef typename boost::shared_ptr<TFAbstractFunction<dim>> Ptr;
	
	virtual TF::Color getMappedRGBfColor(const TF::Size value, const TF::Size dimension) = 0;

	virtual typename Ptr clone() = 0;

	//TF::Color::MapPtr getColorMap();
	
	TF::Size getDimension() const{

		return dim;
	}
	
	TF::MultiDColor<dim>& operator[](const TF::Size index){

		return (*colorMap_)[index];
	}
	
	TF::Size getDomain() const{

		return domain_;
	}
	/*
	TF::MultiDColor<dim>::Map::Ptr getColorMap(){

		return colorMap_;
	}
	*/	
	void clear(){

		TF::MultiDColor<dim>::Map::iterator begin = colorMap_->begin();
		TF::MultiDColor<dim>::Map::iterator end = colorMap_->end();
		for(TF::MultiDColor<dim>::Map::iterator it = begin; it!=end; ++it)
		{
			*it = TF::MultiDColor<dim>();
 		}
	}
	
	void resize(const TF::Size domain){
		
		if(domain == domain_) return;
		domain_ = domain;

		TF::MultiDColor<dim>::Map::Ptr resized(new TF::MultiDColor<dim>::Map(domain_));
		resize_(colorMap_, resized);

		colorMap_ = resized;
	}
	
	void operator=(const TFAbstractFunction<dim> &function){
		
		*colorMap_ = *function.colorMap_;
		domain_ = function.domain_;
	}
	 
	void save(TFXmlWriter::Ptr writer){

		saveSettings_(writer);

		writer->writeStartElement("Function");

			writer->writeAttribute("Domain", TF::convert<TF::Size, std::string>(domain_));
			writer->writeAttribute("Dimension", TF::convert<TF::Size, std::string>(dim));
				
			for(TF::Size i = 0; i < domain_; ++i)
			{
				writer->writeStartElement("MultiDColor");

				for(TF::Size j = 1; j <= dim; ++j)
				{
					writer->writeStartElement("Color");

						writer->writeAttribute("Component1",
							TF::convert<float, std::string>((*colorMap_)[i][j].component1));
						writer->writeAttribute("Component2",
							TF::convert<float, std::string>((*colorMap_)[i][j].component2));
						writer->writeAttribute("Component3",
							TF::convert<float, std::string>((*colorMap_)[i][j].component3));
						writer->writeAttribute("Alpha",
							TF::convert<float, std::string>((*colorMap_)[i][j].alpha));
				
					writer->writeEndElement();
				}

				writer->writeEndElement();
			}

		writer->writeEndElement();
	}

	virtual bool load(TFXmlReader::Ptr reader, bool& sideError){

		#ifndef TF_NDEBUG
			std::cout << "Loading function..." << std::endl;
		#endif
			
		sideError = !loadSettings_(reader);

		bool ok = false;
		if(reader->readElement("Function"))
		{		
			TF::Size domain = TF::convert<std::string, TF::Size>(
				reader->readAttribute("Domain"));

			TF::MultiDColor<dim>::Map::Ptr loaded(new TF::MultiDColor<dim>::Map(domain));			
			TF::Size i = 0;
			for(; i < domain; ++i)
			{
				if(!loadMultiDColor_(reader, loaded, i)) break;
			}

			if(i == domain)
			{
				ok = true;
				resize_(loaded, colorMap_);
			}
		}
		return ok;
	}

protected:

	typename TF::MultiDColor<dim>::Map::Ptr colorMap_;
	TF::Size domain_;

	TFAbstractFunction(){}
	virtual ~TFAbstractFunction(){}

	void resize_(const typename TF::MultiDColor<dim>::Map::Ptr old,
		typename TF::MultiDColor<dim>::Map::Ptr resized){

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
				TF::MultiDColor<dim> computedValue;
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

	virtual void saveSettings_(TFXmlWriter::Ptr writer){}
	virtual bool loadSettings_(TFXmlReader::Ptr reader){ return true; }

	bool loadMultiDColor_(TFXmlReader::Ptr reader, typename TF::MultiDColor<dim>::Map::Ptr loaded,
		TF::Size i){

		bool ok = false;
		if(reader->readElement("MultiDColor"))
		{			
			TF::Size j = 1;	
			for(; j <= dim; ++j)
			{
				if(!loadColor_(reader, loaded, i, j)) break;
			}
			if(j == dim+1)
			{
				ok = true;
			}
		}
		return ok;
	}

	bool loadColor_(TFXmlReader::Ptr reader, typename TF::MultiDColor<dim>::Map::Ptr loaded,
		TF::Size i, TF::Size j){

		bool ok = false;
		if(reader->readElement("Color"))
		{		
			(*loaded)[i][j].component1 = TF::convert<std::string, float>(
				reader->readAttribute("Component1"));
			(*loaded)[i][j].component2 = TF::convert<std::string, float>(
				reader->readAttribute("Component2"));
			(*loaded)[i][j].component3 = TF::convert<std::string, float>(
				reader->readAttribute("Component3"));
			(*loaded)[i][j].alpha = TF::convert<std::string, float>(
				reader->readAttribute("Alpha"));
			ok = true;
		}
		return ok;
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION
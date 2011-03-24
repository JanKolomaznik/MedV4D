#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

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

		const TF::MultiDColor<dim>::Map::Ptr old = colorMap_;
		TF::MultiDColor<dim>::Map::Ptr resized(new TF::MultiDColor<dim>::Map(domain_));

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

		colorMap_ = resized;
	}
	
	void operator=(const TFAbstractFunction<dim> &function){
		
		*colorMap_ = *function.colorMap_;
		domain_ = function.domain_;
	}

	//void save();
	//void load();

protected:

	typename TF::MultiDColor<dim>::Map::Ptr colorMap_;
	TF::Size domain_;

	TFAbstractFunction(){}
	virtual ~TFAbstractFunction(){}
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION
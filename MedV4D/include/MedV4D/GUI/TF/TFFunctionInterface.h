#ifndef TF_FUNCTION_IFACE
#define TF_FUNCTION_IFACE

#include "MedV4D/GUI/TF/TFCommon.h"
#include "MedV4D/GUI/TF/TFColor.h"

namespace M4D {
namespace GUI {

class TFFunctionConstAccessor;

class TFFunctionInterface{

public:

	typedef boost::shared_ptr<TFFunctionInterface> Ptr;
	typedef TFFunctionConstAccessor Const;

	virtual ~TFFunctionInterface(){}

	static const TF::Size defaultDomain = 4096;	//CT

	virtual Ptr clone() = 0;
	
	virtual TF::Color& color(const TF::Coordinates& coords) = 0;

	virtual TF::Color getRGBfColor(const TF::Coordinates& coords) = 0;
	virtual void setRGBfColor(const TF::Coordinates& coords, const TF::Color& value) = 0;

	virtual TF::Size getDomain(const TF::Size dimension) = 0;
	virtual TF::Size getDimension() = 0;

	virtual void resize(const std::vector<TF::Size>& dataStructure) = 0;

	virtual void save(TF::XmlWriterInterface* writer) = 0;
	virtual bool load(TF::XmlReaderInterface* reader) = 0;

protected:

	TFFunctionInterface(){}
};

class TFFunctionConstAccessor{

	friend class TFFunctionInterface;

public:

	const TF::Color nullColor;

	TFFunctionConstAccessor():
		nullColor(-1,-1,-1,-1),
		null_(true){
	}

	TFFunctionConstAccessor(TFFunctionInterface::Ptr function):
		nullColor(-1,-1,-1,-1),
		function_(function),
		null_(false){
	}
		
	TFFunctionConstAccessor(TFFunctionInterface* function):
		nullColor(-1,-1,-1,-1),
		function_(TFFunctionInterface::Ptr(function)),
		null_(false){
	}
	~TFFunctionConstAccessor(){}

	void operator=(const TFFunctionConstAccessor &constPtr){

		function_ = constPtr.function_;
		null_ = constPtr.null_;
	}

	bool operator==(const int &null){

		return null_;
	}

	bool operator!(){

		return null_;
	}

	operator bool(){

		return !null_;
	}
	
	const TF::Color& color(const TF::Coordinates& coords){

		if(null_) return nullColor;
		return function_->color(coords);
	}

	TF::Color getRGBfColor(const TF::Coordinates& coords){

		if(null_) return nullColor;
		return function_->getRGBfColor(coords);
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

	TFFunctionInterface::Ptr function_;
	bool null_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_FUNCTION_IFACE

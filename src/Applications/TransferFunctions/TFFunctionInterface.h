#ifndef TF_FUNCTION_IFACE
#define TF_FUNCTION_IFACE

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

	bool operator!(){

		return null_;
	}

	operator bool(){

		return !null_;
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

} // namespace GUI
} // namespace M4D

#endif //TF_FUNCTION_IFACE
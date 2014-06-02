#ifndef TF_FUNCTION_IFACE
#define TF_FUNCTION_IFACE

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/Color.h"

namespace M4D {
namespace GUI {

class TransferFunctionConstAccessor;

class TransferFunctionInterface {

public:

	typedef boost::shared_ptr<TransferFunctionInterface> Ptr;
	typedef TransferFunctionConstAccessor Const;

	virtual ~TransferFunctionInterface(){}

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

	TransferFunctionInterface(){}
};

class TransferFunctionConstAccessor {

	friend class TransferFunctionInterface;

public:

	const TF::Color nullColor;

	TransferFunctionConstAccessor():
		nullColor(-1,-1,-1,-1),
		null_(true){
	}

	TransferFunctionConstAccessor(TransferFunctionInterface::Ptr function):
		nullColor(-1,-1,-1,-1),
		function_(function),
		null_(false){
	}

	TransferFunctionConstAccessor(TransferFunctionInterface* function):
		nullColor(-1,-1,-1,-1),
		function_(TransferFunctionInterface::Ptr(function)),
		null_(false){
	}
	~TransferFunctionConstAccessor(){}

	void operator=(const TransferFunctionConstAccessor &constPtr){

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

	TransferFunctionInterface::Ptr function_;
	bool null_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_FUNCTION_IFACE

#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <TFTypes.h>

namespace M4D {
namespace GUI {

class TFAbstractFunction{

public:

	typedef boost::shared_ptr<TFAbstractFunction> Ptr;

	virtual ~TFAbstractFunction();

	void operator=(TFAbstractFunction &function);

	TFFunctionType getType() const;

	TFSize getDomain();

	TFColorMapPtr getColorMap();
	
	virtual TFColor getMappedRGBfColor(TFSize value) = 0;

	void clear();

protected:

	TFFunctionType type_;
	TFColorMapPtr colorMap_;

	TFAbstractFunction();
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION
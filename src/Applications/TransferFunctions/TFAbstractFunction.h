#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <TFTypes.h>

namespace M4D {
namespace GUI {

class TFAbstractFunction{

public:

	typedef boost::shared_ptr<TFAbstractFunction> Ptr;

	void operator=(TFAbstractFunction &function);

	TFFunctionType getType() const;

	const TFSize getDomain();

	TFColorMapPtr getColorMap();
	
	virtual TFColor getMappedRGBfColor(const TFSize value) = 0;

	void clear();

	void resize(const TFSize domain);

protected:

	TFFunctionType type_;
	TFColorMapPtr colorMap_;
	TFSize domain_;

	TFAbstractFunction();
	virtual ~TFAbstractFunction();
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION
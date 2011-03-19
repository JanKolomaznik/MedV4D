#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <TFCommon.h>
#include <TFColor.h>

namespace M4D {
namespace GUI {

class TFAbstractFunction{

public:

	typedef boost::shared_ptr<TFAbstractFunction> Ptr;

	static const TF::Size defaultDomain = 4095;	//TODO ?

	void operator=(const TFAbstractFunction &function);
	virtual TFAbstractFunction::Ptr clone() = 0;

	TF::Size getDomain() const;

	//TF::ColorMapPtr getColorMap();
	TF::Color& operator[](const TF::Size index);
	
	virtual TF::Color getMappedRGBfColor(const TF::Size value) = 0;

	//void save();
	//void load();

	void clear();

	void resize(const TF::Size domain);

protected:

	TF::ColorMapPtr colorMap_;
	TF::Size domain_;

	TFAbstractFunction();
	virtual ~TFAbstractFunction();
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION
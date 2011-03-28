#ifndef TF_COLOR
#define TF_COLOR

namespace M4D {
namespace GUI {
namespace TF {

struct Color{	

	typedef std::vector<Color> Map;
	typedef boost::shared_ptr<Map> MapPtr;

	float component1, component2, component3, alpha;

	Color(): component1(0), component2(0), component3(0), alpha(0){}
	Color(const Color &color):
		component1(color.component1),
		component2(color.component2),
		component3(color.component3),
		alpha(color.alpha){
	}
	
	Color(const float component1, const float component2, const float component3, const float alpha):
		component1(component1),
		component2(component2),
		component3(component3),
		alpha(alpha){
	}
	
	Color& operator=(const Color& color){

		component1 = color.component1;
		component2 = color.component2;
		component3 = color.component3;
		alpha = color.alpha;

		return *this;
	}

	bool operator==(const Color& color){
		return (component1 == color.component1) && (component2 == color.component2) && (component3 == color.component3) && (alpha == color.alpha);
	}
	
	bool operator!=(const Color& color){
		return !operator==(color);
	}

	Color operator+(const Color& color){

		return Color(component1 + color.component1,
			component2 + color.component2,
			component3 + color.component3,
			alpha + color.alpha);
	}

	Color operator-(const Color& color){

		return Color(component1 - color.component1,
			component2 - color.component2,
			component3 - color.component3,
			alpha - color.alpha);
	}

	Color operator*(const Color& color){

		return Color(component1 * color.component1,
			component2 * color.component2,
			component3 * color.component3,
			alpha * color.alpha);
	}

	Color operator/(const Color& color){

		return Color(component1 / color.component1,
			component2 / color.component2,
			component3 / color.component3,
			alpha / color.alpha);
	}

	void operator+=(const Color& color){

		component1 += color.component1;
		component2 += color.component2;
		component3 += color.component3;
		alpha += color.alpha;
	}

	void operator-=(const Color& color){

		component1 -= color.component1;
		component2 -= color.component2;
		component3 -= color.component3;
		alpha -= color.alpha;
	}

	void operator*=(const Color& color){

		component1 *= color.component1;
		component2 *= color.component2;
		component3 *= color.component3;
		alpha *= color.alpha;
	}

	void operator/=(const Color& color){

		component1 /= color.component1;
		component2 /= color.component2;
		component3 /= color.component3;
		alpha /= color.alpha;
	}

	Color operator+(const float value){

		return Color(component1 + value,
			component2 + value,
			component3 + value,
			alpha + value);
	}

	Color operator-(const float value){

		return Color(component1 - value,
			component2 - value,
			component3 - value,
			alpha - value);
	}

	Color operator*(const float value){

		return Color(component1 * value,
			component2 * value,
			component3 * value,
			alpha * value);
	}

	Color operator/(const float value){

		return Color(component1 / value,
			component2 / value,
			component3 / value,
			alpha / value);
	}

	void operator+=(const float value){

		component1 += value;
		component2 += value;
		component3 += value;
		alpha += value;
	}

	void operator-=(const float value){

		component1 -= value;
		component2 -= value;
		component3 -= value;
		alpha -= value;
	}

	void operator*=(const float value){

		component1 *= value;
		component2 *= value;
		component3 *= value;
		alpha *= value;
	}

	void operator/=(const float value){

		component1 /= value;
		component2 /= value;
		component3 /= value;
		alpha /= value;
	}
};

template<Size dim>
class MultiDColorVector;

template<Size dim>
class MultiDColor{

public:

	typedef typename boost::shared_ptr<MultiDColor<dim>> Ptr;

	typedef Color* iterator;
	typedef Color* const_iterator;

	typedef Color value_type;

	typedef typename MultiDColorVector<dim> Map;

	MultiDColor(){

		if(dim == 0)
		{
			throw std::exception("Dimension can not be zero");
		}
	}

	MultiDColor(const MultiDColor &color){

		for(Size i = 0; i < dim; ++i) colors_[i] = color.colors_[i];
	}

	MultiDColor(const float component1, const float component2, const float component3, const float alpha){

		for(Size i = 0; i < dim; ++i) colors_[i] = Color(component1, component2, component3, alpha);
	}

	Size getDimension(){

		return dim;
	}

	iterator begin(){

		return colors_;
	}

	iterator end(){

		return colors_ + dim;
	}

	const_iterator begin() const{

		return colors_;
	}

	const_iterator end() const{

		return colors_ + dim;
	}

	void operator=(const MultiDColor& color){

		for(Size i = 0; i < dim; ++i) colors_[i] = color.colors_[i];
	}

	value_type& operator[](const Size dimension){

		if(dimension > dim || dimension == 0)
		{
			throw std::out_of_range("Wrong dimension");
		}

		return colors_[dimension-1];
	}

	bool operator==(const MultiDColor& color){

		bool equal = true;
		for(Size i = 0; i < dim; ++i) if(colors_[i] != color.colors_[i]) equal = false;

		return equal;
	}

	bool operator!=(const MultiDColor& color){

		return !operator==(color);
	}

	void operator+=(const MultiDColor& color){

		for(Size i = 0; i < dim; ++i) colors_[i] += color.colors_[i];
	}

	void operator-=(const MultiDColor& color){

		for(Size i = 0; i < dim; ++i) colors_[i] -= color.colors_[i];
	}

	void operator*=(const MultiDColor& color){

		for(Size i = 0; i < dim; ++i) colors_[i] *= color.colors_[i];
	}

	void operator/=(const MultiDColor& color){

		for(Size i = 0; i < dim; ++i) colors_[i] /= color.colors_[i];
	}

	MultiDColor operator+(const MultiDColor& color){

		MultiDColor result(*this);
		result += color;
		return result;
	}

	MultiDColor operator-(const MultiDColor& color){

		MultiDColor result(*this);
		result -= color;
		return result;
	}

	MultiDColor operator*(const MultiDColor& color){

		MultiDColor result(*this);
		result *= color;
		return result;
	}

	MultiDColor operator/(const MultiDColor& color){

		MultiDColor result(*this);
		result /= color;
		return result;
	}

	void operator+=(const float value){

		for(Size i = 0; i < dim; ++i) colors_[i] += value;
	}

	void operator-=(const float value){

		for(Size i = 0; i < dim; ++i) colors_[i] -= value;
	}

	void operator*=(const float value){

		for(Size i = 0; i < dim; ++i) colors_[i] *= value;
	}

	void operator/=(const float value){

		for(Size i = 0; i < dim; ++i) colors_[i] /= value;
	}

	MultiDColor operator+(const float value){

		MultiDColor result(*this);
		result += value;
		return result;
	}

	MultiDColor operator-(const float value){

		MultiDColor result(*this);
		result -= value;
		return result;
	}

	MultiDColor operator*(const float value){

		MultiDColor result(*this);
		result *= value;
		return result;
	}

	MultiDColor operator/(const float value){

		MultiDColor result(*this);
		result /= value;
		return result;
	}

private:

	Color colors_[dim];
};

template<Size dim>
class MultiDColorVector{

public:

	typedef typename boost::shared_ptr<MultiDColorVector<dim>> Ptr;

	typedef typename MultiDColor<dim> value_type;

	typedef typename std::vector<value_type>::iterator iterator;
	typedef typename std::vector<value_type>::const_iterator const_iterator;

	MultiDColorVector(const Size size):
		colors_(size, MultiDColor<dim>()){
	}

	void operator=(const MultiDColorVector& colorVector){

		colors_.clear();

		const_iterator begin = colorVector.begin();
		const_iterator end = colorVector.end();
		for(const_iterator it = begin; it!=end; ++it)
		{
			colors_.push_back(*it);
		}
	}

	Size size(){

		return colors_.size();
	}

	value_type& operator[](const Size index){

		return colors_[index];
	}

	iterator begin(){

		return colors_.begin();
	}

	iterator end(){

		return colors_.end();
	}

	const_iterator begin() const{

		return colors_.begin();
	}

	const_iterator end() const{

		return colors_.end();
	}

private:

	std::vector<MultiDColor<dim>> colors_;
};

}
}
}

#endif	//TF_COLOR
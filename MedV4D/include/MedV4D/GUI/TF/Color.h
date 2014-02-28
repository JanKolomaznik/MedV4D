#ifndef TF_COLOR
#define TF_COLOR

#include "MedV4D/GUI/TF/Common.h"

namespace M4D {
namespace GUI {
namespace TF {

struct Color{	

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

}
}
}

#endif	//TF_COLOR

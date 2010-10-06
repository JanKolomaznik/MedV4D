/***********************************
	CGLOBALS
************************************/

#include <string>
#include <math.h>

#include "globals.h"

namespace viewer {

float vectSize(const SPoint3D<float> &vect) { 
	return sqrt(vect.x * vect.x + vect.y * vect.y + vect.z * vect.z); 
}

float vectDot(const SPoint3D<float> &v1, const SPoint3D<float> &v2) { 
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; 
}

/*std::string wcs2str(const std::wstring &wstr) {
	std::string retval;
	char *buff;
	buff = new char[wstr.size()+1];
	wcstombs(buff, wstr.c_str(), wstr.size()+1);

	retval = buff;
	delete buff;
	return retval;
}

std::wstring str2wcs(const std::string &str) {
	std::wstring retval;
	wchar_t *buff;
	buff = new wchar_t[str.size()+1];
	mbstowcs(buff, str.c_str(), str.size()+1);

	retval = buff;
	delete buff;
	return retval;
}
*/
double exp_vals[10000];

// gets numbers from -50 to 50
void exp_fast_prepare() {
	for(int i = 0; i < 10000; i++) {
		exp_vals[i] = exp((double)(i - 5000) / 100);
	}
}

// gets numbers from -100 to 100
double exp_fast(double param) {
	return exp_vals[(int)(param * 100 + 5000)];
}

}
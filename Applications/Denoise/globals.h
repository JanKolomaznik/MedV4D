/***********************************
	CGLOBALS
************************************/
 
namespace viewer {

#define EPSILON		0.00001

#define DEGTORAD(deg)	((M_PI*(deg))/180.0)
#define RADTODEG(rad)	((180.0*(rad))/M_PI)



class CVector4;

template<class T>
struct SPoint2D {
	T x;
	T y;

	void init(T nx, T ny) { x = nx; y = ny; }
	bool operator ==(const SPoint2D<T> &pt2) const { return (x == pt2.x) && (y == pt2.y); }
	bool operator !=(const SPoint2D<T> &pt2) const { return (x != pt2.x) || (y != pt2.y); }

	SPoint2D() {};
	SPoint2D(T x, T y) : x(x), y(y) {};
};

template<class T>
struct SPoint3D {
	T x;
	T y;
	T z;

	void init(T nx, T ny, T nz) { x = nx; y = ny; z = nz; }
	bool operator==(const SPoint3D<T> &b) const { return ((x == b.x) && (y == b.y) && (z == b.z)); }
	SPoint3D operator-(const SPoint3D<T> &b) const { return SPoint3D<T>(x - b.x, y - b.y, z - b.z); }
	SPoint3D operator+(const SPoint3D<T> &b) const { return SPoint3D<T>(x + b.x, y + b.y, z + b.z); }
	T dot(const SPoint3D<T> &v2) { return x * v2.x + y * v2.y + z * v2.z; }
//	SPoint3D<T>& operator=(CVector4 &vec) {x = vec.x; y = vec.y; z = vec.z; return *this;}

	SPoint3D() {};
	SPoint3D(T x, T y, T z) : x(x), y(y), z(z) {};
};

template <class T>
class CArray {
	T *pData;
	int size;

public:
	T* alloc(int newSize) {reset(); pData = new T[newSize]; size = (pData != NULL)?newSize:0; return pData; }
	void reset() { if(pData) delete[] pData; pData = NULL; size = 0; }
	inline T* ptr() { return pData;}
	int getSize() { return size;}
	inline T& operator[](unsigned int idx) { return pData[idx]; }

	CArray() : pData(NULL), size(0) {}
	~CArray() {reset();}
};

float vectSize(const SPoint3D<float> &vect);
float vectDot(const SPoint3D<float> &v1, const SPoint3D<float> &v2);
/*
// converts wstring into string
std::string wcs2str(const std::wstring &wstr);

// converts string into wstring
std::wstring str2wcs(const std::string &str);
*/
void exp_fast_prepare();

double exp_fast(double param);

}

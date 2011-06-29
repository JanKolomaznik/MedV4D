/***********************************
	CVOLUMESET
************************************/

//#define MACINTOSH

#include <omp.h>
#include <algorithm>
#include <queue>
#include <cstring>
#include <ctime>

#ifdef OPENCL

#include "MyOpenCL.h"

#endif

#include "common/Common.h"

namespace viewer {

class CInfoDialog;
class CProgress;

template <class T>
class CVolumeSet;

template<class T>
bool volWatershedBasic(CVolumeSet<T> &dest, const CVolumeSet<T> &src, const CVolumeSet<int> &markers);

template <class T2>
bool volGetLocalMinima(const CVolumeSet<T2> &src, CVolumeSet<int> &markers);

// converter functions for copying/converting volumesets between different types
template <class D, class S>
inline void  inlineConvert(D &dst, const S &src) {
	dst = (D) src;
}

template <>
inline void inlineConvert<short, float>(short &dst, const float &src) {
	if(src > 32767)
		dst = 32767;
	else if(src < -32768)
		dst = -32768;
	else
		dst = (short)(src);
}

template <>
inline void inlineConvert<unsigned char, float>(unsigned char &dst, const float &src) {
	if(src > 255)
		dst = 255;
	else if(src < 0)
		dst = 0;
	else
		dst = (unsigned char)(src);
}

// computes a L2 norm between two vectors
template <class T>
inline T L2Norm(const std::vector<T> &vec1, const std::vector<T> &vec2) {
	int i, imax;
	imax = (int)std::min(vec1.size(), vec2.size());
	double accum = 0;
	for(i = 0; i < imax; i++) {
		double val = (double) vec2[i] - (double) vec1[i];
		accum += val * val;
	}
	return (T) sqrt(accum);
}

// computes a L2 norm between two vectors
template <class T>
inline T L2Norm(const T *vec1, const T *vec2, unsigned int size) {
	unsigned int i;
	double accum = 0;
	for(i = 0; i < size; i++) {
		double val = (double) vec2[i] - (double) vec1[i];
		accum += val * val;
	}
	return (T) sqrt(accum);
}


// computes a squared L2 norm between two vectors
template <class T>
inline T L2NormSq(const std::vector<T> &vec1, const std::vector<T> &vec2) {
	int i, imax;
	imax = (int)std::min(vec1.size(), vec2.size());
	double accum = 0;
	for(i = 0; i < imax; i++) {
		double val = (double) vec2[i] - (double) vec1[i];
		accum += val * val;
	}
	return (T) (accum);
}

// computes a squared L2 norm between two vectors
template <class T>
inline T L2NormSq(const T *vec1, const T *vec2, unsigned int size) {
	unsigned int i;
	double accum = 0;
	for(i = 0; i < size; i++) {
		double val = (double) vec2[i] - (double) vec1[i];
		accum += val * val;
	}
	return (T) (accum);
}


// returns the sum of all vector members
template <class T>
T getVectorSum(const std::vector<T> &v) {
	T vectorSum = 0;
	for(int i = 0; i < (int)v.size(); i++) {
		vectorSum += v[i];
	}
	return vectorSum;
}

template <class T>
class CVolumeSet {
public:
	enum TCutTypes { cutTypeXY, cutTypeXZ, cutTypeYZ };
protected:
	CVolumeSet();
	CVolumeSet(CVolumeSet& other) {};
	int width, height, depth;
	SPoint3D<int> origPos;

public:
	std::vector<T*> planes;
protected:
	T *cutXZ, *cutYZ;

	void reset() {
		int i;
		for (i = 0; i < (int)planes.size(); i++) {
			delete[] planes[i];
		}
		planes.clear();
		width = height = depth = 0;
		// reset cuts through planes
		if(cutXZ)
			delete[] cutXZ;
		cutXZ = NULL;
		if(cutYZ)
			delete[] cutYZ;
		cutYZ = NULL;
	};

	CVolumeSet<T> &operator=(CVolumeSet<T> &other) {};
public:
	inline int getWidth() const { return width; }
	inline int getHeight() const { return height; }
	inline int getDepth() const { return depth; }
	void getSize(SPoint3D<int> &size) const { size.x = width; size.y = height; size.z = depth; }
	SPoint3D<int> getSize() const { return SPoint3D<int> (width, height, depth); }
	void getSize(int &iWidth, int &iHeight, int &iDepth) const {iWidth = width; iHeight = height; iDepth = depth; }
	int getOrigPosX() const { return origPos.x; }
	int getOrigPosY() const { return origPos.y; }
	int getOrigPosZ() const { return origPos.z; }
	SPoint3D<int> getOrigPos() const { return origPos; }
	void setOrigPos(int posX, int posY, int posZ) { origPos.x = posX; origPos.y = posY; origPos.z = posZ; }
	bool isPtInVolume(int x, int y, int z, bool globalCoords = true) {
		if(globalCoords) {
			x -= origPos.x;
			y -= origPos.y;
			z -= origPos.z;
		}
		return((x >= 0) && (x < width) && (y >= 0) && (y < height) && (z >= 0) && (z < depth));
	}

	bool realloc(unsigned int newX, unsigned int newY, unsigned int newZ) {
		reset();

		width = newX;
		height = newY;
		depth = newZ;

		// allocate main volume
		planes.resize(depth);
		int i;
		for (i = 0; i < depth; i++) {
			planes[i] = new T[width * height];
		}

		// allocate memory for cuts through planes
		cutXZ = new T[width * depth];
		cutYZ = new T[height * depth];

		return true;
	};

	// saves volumeset to disk
	bool saveToDisk(const char *wsFilename, CInfoDialog *info) {
		FILE *fw;
		if(NULL == (fw = fopen(wsFilename, "wb"))) {
/*			std::wstring errstr;
			errstr = std::wstring(L"Could not create file: ") + std::wstring(wsFilename);
			if(info)
				info->setMessage(errstr);*/
			return false;
		}

		unsigned char header[3];
		header[0] = 'V';
		header[1] = 'o';
		header[2] = 'l';
		fwrite(header, sizeof(unsigned char), 3, fw);

		unsigned int size[3];
		size[0] = width;
		size[1] = height;
		size[2] = depth;
		fwrite(size, sizeof(unsigned int), 3, fw);

		const char *name = typeid(T).name();
		char buff[32];
		// HACK: for macintosh
		if(name[0] == 'f')
			strncpy(buff, "float", 31);
		else
			strncpy(buff, name,31);
		fwrite(buff, sizeof(char), 32, fw);
		
		for(int k = 0; k < depth; k++)
		{
			fwrite(planes[k], sizeof(T), width*height, fw);
		}

		fclose(fw);
		return true;
	}

	// loads volumeset from disk
	void loadFromDisk(const char *wsFilename, CInfoDialog *info) {
		FILE *fr;
		if(NULL == (fr = fopen(wsFilename, "rb"))) {
/*			std::wstring errstr;
			errstr = std::wstring(L"Could not open file: ") + std::wstring(wsFilename);
			if(info)
				info->setMessage(errstr);*/
			return;
		}

		unsigned char header[3];
		fread(header, sizeof(unsigned char), 3, fr);
		if(header[0] != 'V' || header[1] != 'o' || header[2] != 'l')
		{
/*			std::wstring errstr;
			errstr = std::wstring(L"Wrong file type: ") + std::wstring(wsFilename);
			if(info)
				info->setMessage(errstr);
			fclose(fr);*/
			return;
		}

		unsigned int size[3];
		fread(size, sizeof(unsigned int), 3, fr);
		width = size[0];
		height = size[1];
		depth = size[2];

		char name[32];
		fread(name, sizeof(char), 32, fr);
		if(0 != strcmp(name, typeid(T).name()) && name[0] != typeid(T).name()[0])
		{
/*			std::wstring errstr;
			errstr = std::wstring(L"Wrong file type (") + str2wcs(typeid(T).name()) + L"): " + wsFilename + L"(" + str2wcs(name) + L")";
			if(info)
				info->setMessage(errstr);*/
			fclose(fr);
			return;
		}

		this->realloc(width, height, depth);
		
		for(int k = 0; k < depth; k++)
		{
			fread(planes[k], sizeof(T), width*height, fr);
		}

		fclose(fr);
	}

	// makes this volumeset the same size as different volumeset
	template <class T2>
	void copySize(const CVolumeSet<T2> &sourceVolume) {
		if((getDepth() != sourceVolume.getDepth()) ||
			(getHeight() != sourceVolume.getHeight()) ||
			(getWidth() != sourceVolume.getWidth())) {
			realloc(sourceVolume.getWidth(), sourceVolume.getHeight(), sourceVolume.getDepth());
		}
	}
	// copies original position of volumeset
	template <class T2>
	void copyOrigPos(const CVolumeSet<T2> &sourceVolume) {
		origPos.x = sourceVolume.getOrigPosX();
		origPos.y = sourceVolume.getOrigPosY();
		origPos.z = sourceVolume.getOrigPosZ();
	}

	// returns position that lies at a position in original coordinates, if that is outside the volumeset, then outsideValue is returned
	T getOriginalPosValue(int x, int y, int z, T outsideValue) const {
		int volumePosX = x - origPos.x;
		int volumePosY = y - origPos.y;
		int volumePosZ = z - origPos.z;
		if ((volumePosX >= 0) && (volumePosX < width) && 
			(volumePosY >= 0) && (volumePosY < height) && 
			(volumePosZ >= 0) && (volumePosZ < depth)) {
				return planes[volumePosZ][volumePosY * width + volumePosX];
		}else
			return outsideValue;
	}

	//clears the whole volume with given value
	void setValue(T value){
		int imgSize = width * height;
#pragma omp parallel for
		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < imgSize; j++)
				planes[i][j] = value;
		}
	}

	//clears the whole volume with zero (only OK with types convertible to 0)
	void setZero(){
		int imgSize = width * height * sizeof(T);
#pragma omp parallel for
		for (int i = 0; i < depth; i++) {
			memset(planes[i], 0, imgSize);
		}
	}

	// returns value at specified position
	T getValue(unsigned int x, unsigned int y, unsigned int z) const{
		assert(((int)x < width) && ((int)y < height) && ((int)z < depth));
		return planes[z][y * width + x];
	}

	// returns value at specified position
	void setValue(unsigned int x, unsigned int y, unsigned int z, T newValue){
		assert(((int)x < width) && ((int)y < height) && ((int)z < depth));
		planes[z][y * width + x] = newValue;
	}

	//returns the maximal value in the volume
	T getMaxValue(){
		int imgSize = width * height;
		T maxval = planes[0][0];
		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < imgSize; j++)
				if(planes[i][j] > maxval)
					maxval = planes[i][j];
		}
		return maxval;
	}

	// returns pointer to a plane with one XY image
	inline const T* getXYPlane(unsigned int z) const { return planes[z]; };

	// returns pointer to a plane with one XY image, not constant, so it can be written
	inline T* getXYPlaneNonconst(unsigned int z) { return planes[z]; };

	// returns pointer to a plane with one XZ image
	const T* getXZPlane(unsigned int y) const { 
		int i, j;
		// TODO: optimize
		for(j = 0; j < depth; j++) {
			for(i = 0; i < width; i++) {
				cutXZ[j * width + i] = planes[j][y * width + i];
			}
		}

		return cutXZ;
	};

	// returns pointer to a plane with one XZ image
	const T* getYZPlane(unsigned int x) const { 
		// TODO: optimize
		for(int j = 0; j < depth; j++) {
			for(int i = 0; i < height; i++) {
				cutYZ[j * height + i] = planes[j][i * width + x];
			}
		}

		return cutYZ;
	};

	// debug: checks if all voxels are nonnegative
	bool dbgCheckForNegativeValues() {
		for(int i = 0; i < width; i++) {
			for(int j = 0; j < height; j++) {
				for(int k = 0; k < depth; k++) {
					if(getValue(i, j, k) < 0) {
						assert(0);
						return false;
					}
				}
			}
		}	
		return true;
	}

	long int getNrVoxels() {
		return (long int)width * (long int)height * (long int)depth;
	}

	// for all voxels below threshold set mask to 1
	void threshold(CVolumeSet<unsigned char> &mask, T threshold) {
		mask.copySize(*this);
		mask.copyOrigPos(*this);

		SPoint3D<int> srcSize = getSize();
#pragma omp parallel for
		for(int k = 0; k < srcSize.z; k++) {
			int total = srcSize.x * srcSize.y;
			const T* srcData = getXYPlane(k);
			unsigned char* maskData = mask.getXYPlaneNonconst(k);
			for(int i = 0; i < total; i++) {
				maskData[i] = (srcData[i] <= threshold)?1:0;
			}
		}
	}

	template <class T2>
	void copyVolume(const CVolumeSet<T2> &src) {
		int width, height, depth;
		width = src.getWidth();
		height = src.getHeight();
		depth = src.getDepth();

		copySize(src);

		int totalPelsPerPlane = width * height;
#pragma omp parallel for
		for(int i = 0; i < depth; i++) {
			const T2 *srcPlane = src.getXYPlane(i);
			T *destPlane = getXYPlaneNonconst(i);
			for(int j = 0; j < totalPelsPerPlane; j++) {
				inlineConvert<T, T2>(destPlane[j],srcPlane[j]);
			}
		}
	}

	// performs linear combination on two datasets dest = a * src1 + b * src2 + c
	void linearCombination(const CVolumeSet<T> &src1, const CVolumeSet<T> &src2, T a, T b, T c) {
		copySize(src1);
		assert(src1.getSize() == src2.getSize());
		if(!(src1.getSize() == src2.getSize()))
			return;

		SPoint3D<int> srcSize = src1.getSize();
#pragma omp parallel for
		for(int k = 0; k < srcSize.z; k++) {
			int total = srcSize.x * srcSize.y;
			const T* src1Data = src1.getXYPlane(k);
			const T* src2Data = src2.getXYPlane(k);
			T* destData = getXYPlaneNonconst(k);
			for(int i = 0; i < total; i++) {
				destData[i] = a * src1Data[i] + b * src2Data[i] + c;
			}
		}
	}

	// multiplies two volumesets voxel-wise (res = A * B)
	void multiply(const CVolumeSet<T> &volA, const CVolumeSet<T> &volB) {
		if(!(volA.getSize() == volB.getSize()))
			return;
		copySize(volA);
		SPoint3D<int> srcSize;
		volA.getSize(srcSize);
#pragma omp parallel for
		for(int k = 0; k < srcSize.z; k++) {
			int total = srcSize.x * srcSize.y;
			const T* srcData1 = volA.getXYPlane(k);
			const T* srcData2 = volB.getXYPlane(k);
			T* destData = getXYPlaneNonconst(k);
			for(int i = 0; i < total; i++) {
				destData[i] = srcData1[i] * srcData2[i];
			}
		}
	}

	// multiplies volumeset by a constant value
	void multiply(T value) {
		int imgSize = width * height;
#pragma omp parallel for
		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < imgSize; j++)
				planes[i][j] *= value;
		}
	}

	// subtracts one volume from another (res = A - B)
	void subtract(const CVolumeSet<T> &volA, const CVolumeSet<T> &volB) {
		if(!(volA.getSize() == volB.getSize()))
			return;
		copySize(volA);
		SPoint3D<int> srcSize;
		volA.getSize(srcSize);
#pragma omp parallel for
		for(int k = 0; k < srcSize.z; k++) {
			int total = srcSize.x * srcSize.y;
			const T* srcData1 = volA.getXYPlane(k);
			const T* srcData2 = volB.getXYPlane(k);
			T* destData = getXYPlane(k);
			for(int i = 0; i < total; i++) {
				destData[i] = srcData1[i] - srcData2[i];
			}
		}
	}
	
	// adds an offset to all voxels
	void addOffset(T offset) {
		int imgSize = width * height;
#pragma omp parallel for
		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < imgSize; j++)
				planes[i][j] += offset;
		}
	}



	// takes volume, all values below MIN will become LOWVALUE, all values abovee MAX becomes HIGHVALUE, all values inbetween are linearly stretched between LOWVALUE and HIGHVALUE
	// CAUTION: not safe for small data types, such as char
	void volClampAndStretch(const CVolumeSet<T> &src, T minValue, T maxValue, T lowValue, T highValue) {
		copySize(src);
		SPoint3D<int> srcSize;
		src.getSize(srcSize);
#pragma omp parallel for
		for(int k = 0; k < srcSize.z; k++) {
			int total = srcSize.x * srcSize.y;
			const T* srcData = src.getXYPlane(k);
			T* destData = getXYPlaneNonconst(k);
			for(int i = 0; i < total; i++) {
				T value = srcData[i];
				if(value < minValue)
					destData[i] = lowValue;
				else if(value > maxValue)
					destData[i] = highValue;
				else 
					destData[i] = lowValue + (value - minValue) * (highValue - lowValue) / (maxValue - minValue);
			}
		}
	}

	// computes binary floodfill for all pixels that are 0s and are bordered by 1s in one plane in volumeset
	// caution: this method fills only one image in volumeset (the one where seed lies)
	void volTreshold2DFloodfill(SPoint3D<unsigned int> seed) {
		CVolumeSet<unsigned char> *ONLY_FOR_UCHAR_VOLUMESET = this;

		std::list<SPoint2D<unsigned int> > queue;
		queue.clear();

		// initialize 
		unsigned char srcValue = getValue(seed.x, seed.y, seed.z);
		if((srcValue == 0)) {
			queue.push_back(SPoint2D<unsigned int>(seed.x, seed.y));
		}

		while(!queue.empty()) {
			unsigned char value;
			SPoint2D<unsigned int> activePt = *(queue.begin());
			queue.pop_front();

			//left neighbour
			if((int)activePt.x > 0) { 
				value = getValue(activePt.x - 1, activePt.y, seed.z);
				if(value == 0 ) { // not visited yet
					queue.push_back(SPoint2D<unsigned int>(activePt.x - 1, activePt.y));
					setValue(activePt.x - 1, activePt.y, seed.z, 1);
				}
			}
			// right neighbour
			if((int)activePt.x < getWidth() - 1) {
				value = getValue(activePt.x + 1, activePt.y, seed.z);
				if(value == 0) { // not visited yet
					queue.push_back(SPoint2D<unsigned int>(activePt.x + 1, activePt.y));
					setValue(activePt.x + 1, activePt.y, seed.z, 1);
				}
			}
			// upper neighbour
			if((int)activePt.y > 0) {
				value = getValue(activePt.x, activePt.y - 1, seed.z);
				if(value == 0) { // not visited yet
					queue.push_back(SPoint2D<unsigned int>(activePt.x, activePt.y - 1));
					setValue(activePt.x, activePt.y - 1, seed.z, 1);
				}
			}
			// lower neighbour
			if((int)activePt.y < getHeight() - 1) {
				value = getValue(activePt.x, activePt.y + 1, seed.z);
				if(value == 0) { // not visited yet
					queue.push_back(SPoint2D<unsigned int>(activePt.x, activePt.y + 1));
					setValue(activePt.x, activePt.y + 1, seed.z, 1);
				}
			}
		}
	}

	// computes floodfill for all pixels lying between two specified values in one plane in volumeset
	// caution: this method fills only one image in volumeset (the one where seed lies)
	void volTreshold2DFloodfill(const CVolumeSet<T> &srcVolume, T minValue, T maxValue, SPoint3D<unsigned int> seed) {
		CVolumeSet<unsigned char> *ONLY_FOR_UCHAR_VOLUMESET = this;
		copySize(srcVolume);
		copyOrigPos(srcVolume);
		setZero();

		std::list<SPoint2D<unsigned int> > queue;
		queue.clear();

		// initialize 
		T srcValue = srcVolume.getValue(seed.x, seed.y, seed.z);
		if((srcValue >= minValue) && (srcValue <= maxValue)) {
			queue.push_back(SPoint2D<unsigned int>(seed.x, seed.y));
			setValue(seed.x, seed.y, seed.z, 1);
		}

		while(!queue.empty()) {
			T value;
			SPoint2D<unsigned int> activePt = *(queue.begin());
			queue.pop_front();

			//left neighbour
			if((int)activePt.x > 0) { 
				if(getValue(activePt.x - 1, activePt.y, seed.z) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x - 1, activePt.y, seed.z);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint2D<unsigned int>(activePt.x - 1, activePt.y));
						setValue(activePt.x - 1, activePt.y, seed.z, 1);
					}
				}
			}
			// right neighbour
			if((int)activePt.x < srcVolume.getWidth() - 1) {
				if(getValue(activePt.x + 1, activePt.y, seed.z) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x + 1, activePt.y, seed.z);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint2D<unsigned int>(activePt.x + 1, activePt.y));
						setValue(activePt.x + 1, activePt.y, seed.z, 1);
					}
				}
			}
			// upper neighbour
			if((int)activePt.y > 0) {
				if(getValue(activePt.x , activePt.y - 1, seed.z) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x, activePt.y - 1, seed.z);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint2D<unsigned int>(activePt.x, activePt.y - 1));
						setValue(activePt.x, activePt.y - 1, seed.z, 1);
					}
				}
			}
			// lower neighbour
			if((int)activePt.y < srcVolume.getHeight() - 1) {
				if(getValue(activePt.x , activePt.y + 1, seed.z) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x, activePt.y + 1, seed.z);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint2D<unsigned int>(activePt.x, activePt.y + 1));
						setValue(activePt.x, activePt.y + 1, seed.z, 1);
					}
				}
			}

		}
	}


	// computes floodfill for all pixels lying between two specified values in the whole volumeset
	void volTreshold3DFloodfill(const CVolumeSet<T> &srcVolume, T minValue, T maxValue, SPoint3D<unsigned int> seed) {
		CVolumeSet<unsigned char> *ONLY_FOR_UCHAR_VOLUMESET = this;
		copySize(srcVolume);
		copyOrigPos(srcVolume);
		setZero();
 
		std::list<SPoint3D<unsigned int> > queue;
		queue.clear();

		// initialize 
		T srcValue = srcVolume.getValue(seed.x, seed.y, seed.z);
		if((srcValue >= minValue) && (srcValue <= maxValue)) {
			queue.push_back(SPoint3D<unsigned int>(seed.x, seed.y, seed.z));
			setValue(seed.x, seed.y, seed.z, 1);
		}

		while(!queue.empty()) {
			T value;
			SPoint3D<unsigned int> activePt = *(queue.begin());
			queue.pop_front();

			//left neighbour
			if((int)activePt.x > 0) { 
				if(getValue(activePt.x - 1, activePt.y, activePt.z) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x - 1, activePt.y, activePt.z);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint3D<unsigned int>(activePt.x - 1, activePt.y, activePt.z));
						setValue(activePt.x - 1, activePt.y, activePt.z, 1);
					}
				}
			}
			// right neighbour
			if((int)activePt.x < srcVolume.getWidth() - 1) {
				if(getValue(activePt.x + 1, activePt.y, activePt.z) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x + 1, activePt.y, activePt.z);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint3D<unsigned int>(activePt.x + 1, activePt.y, activePt.z));
						setValue(activePt.x + 1, activePt.y, activePt.z, 1);
					}
				}
			}
			// upper neighbour
			if((int)activePt.y > 0) {
				if(getValue(activePt.x , activePt.y - 1, activePt.z) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x, activePt.y - 1, activePt.z);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint3D<unsigned int>(activePt.x, activePt.y - 1, activePt.z));
						setValue(activePt.x, activePt.y - 1, activePt.z, 1);
					}
				}
			}
			// lower neighbour
			if((int)activePt.y < srcVolume.getHeight() - 1) {
				if(getValue(activePt.x , activePt.y + 1, activePt.z) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x, activePt.y + 1, activePt.z);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint3D<unsigned int>(activePt.x, activePt.y + 1, activePt.z));
						setValue(activePt.x, activePt.y + 1, activePt.z, 1);
					}
				}
			}
			// back neighbour
			if((int)activePt.z > 0) {
				if(getValue(activePt.x , activePt.y, activePt.z - 1) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x, activePt.y, activePt.z - 1);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint3D<unsigned int>(activePt.x, activePt.y, activePt.z - 1));
						setValue(activePt.x, activePt.y, activePt.z - 1, 1);
					}
				}
			}
			// front neighbour
			if((int)activePt.z < srcVolume.getDepth() - 1) {
				if(getValue(activePt.x , activePt.y, activePt.z + 1) == 0) { // not visited yet
					value = srcVolume.getValue(activePt.x, activePt.y, activePt.z + 1);
					if((value >= minValue) && (value <= maxValue)) {	// if value is in borders
						queue.push_back(SPoint3D<unsigned int>(activePt.x, activePt.y, activePt.z + 1));
						setValue(activePt.x, activePt.y, activePt.z + 1, 1);
					}
				}
			}

		}
	}

	// erodes mask by one pixel in 4-neighbourhood
	void erodeMask4() {
		CVolumeSet<unsigned char> *ONLY_FOR_UCHAR_VOLUMESET = this;

		CVolumeSet<unsigned char> srcMask(1,1,1);
		srcMask.copyVolume(*this);

		int total;
		int i, j, k;
		int width, height, depth;
		width = getWidth();
		height = getHeight();
		depth = getDepth();

		total = width * height;
		for(k = 0; k < depth; k++) {
			unsigned char *srcPlane = srcMask.getXYPlaneNonconst(k), *destPlane = getXYPlaneNonconst(k);
			unsigned char *lowerPlane, *higherPlane;
			if(k > 0)
				lowerPlane = srcMask.getXYPlaneNonconst(k - 1);
			else
				lowerPlane = NULL;
			if(k < depth - 1)
				higherPlane = srcMask.getXYPlaneNonconst(k + 1);
			else
				higherPlane = NULL;

			i = 0; j = 0;
			int index;
			for(index = 0; index < total; index++) {
				if(srcPlane[index] != 0) {
					if((i == 0) || (j == 0) || (k == 0) || (i == width - 1) || (j == height - 1) || (k == depth - 1)) {
						destPlane[index] = 0;
					}else {
						bool ok = true;
						// neighbour left/right
						if((srcPlane[index - 1] == 0) || (srcPlane[index + 1] == 0))
							ok = false;
						// neighbour up/down
						if((srcPlane[index - width] == 0) || (srcPlane[index + width] == 0))
							ok = false;
						// neighbour above/below
						if((higherPlane[index] == 0) || (lowerPlane[index] == 0))
							ok = false;
						if(!ok)
							destPlane[index] = 0;
					}

				}
				i++;
				if(i >= width) {
					j++;
					i = 0;
				}
			}
		}
	}

	// dilates mask by one pixel in 4-neighbourhood
	void dilateMask4() {
		CVolumeSet<unsigned char> *ONLY_FOR_UCHAR_VOLUMESET = this;

		CVolumeSet<unsigned char> srcMask(1,1,1);
		srcMask.copyVolume(*this);

		int total;
		int i, j, k;
		int width, height, depth;
		width = getWidth();
		height = getHeight();
		depth = getDepth();

		total = width * height;
		for(k = 0; k < depth; k++) {
			unsigned char *srcPlane = srcMask.getXYPlaneNonconst(k), *destPlane = getXYPlaneNonconst(k);
			unsigned char *lowerPlane, *higherPlane;
			if(k > 0)
				lowerPlane = srcMask.getXYPlaneNonconst(k - 1);
			else
				lowerPlane = NULL;
			if(k < depth - 1)
				higherPlane = srcMask.getXYPlaneNonconst(k + 1);
			else
				higherPlane = NULL;

			i = 0; j = 0;
			int index;
			for(index = 0; index < total; index++) {
				if(srcPlane[index] == 0) {
					bool lookMX, lookPX, lookMY, lookPY, lookMZ, lookPZ;
					lookMX = lookPX = lookMY = lookPY = lookMZ = lookPZ = true;
					if(i == 0)
						lookMX = false;
					if(i == width - 1)
						lookPX = false;
					if(j == 0)
						lookMY = false;
					if(j == height - 1)
						lookPY = false;
					if(k == 0)
						lookMZ = false;
					if(k == depth - 1)
						lookPZ = false;


					bool ok = false;
					// neighbour left/right
					if(lookMX)
						if(srcPlane[index-1] != 0)
							ok = true;
					if(lookPX)
						if(srcPlane[index+1] != 0)
							ok = true;
					if(lookMY)
						if(srcPlane[index - width] != 0)
							ok = true;
					if(lookPY)
						if(srcPlane[index + width] != 0)
							ok = true;
					if(lookMZ)
						if(lowerPlane[index] != 0)
							ok = true;
					if(lookPZ)
						if(higherPlane[index] != 0)
							ok = true;

	//				if((srcPlane[index - 1] == 0) || (srcPlane[index + 1] == 0))
	//					ok = true;
					// neighbour up/down
	//				if((srcPlane[index - width] == 0) || (srcPlane[index + width] == 0))
	//					ok = true;
					// neighbour above/below
	//				if((higherPlane[index] == 0) || (lowerPlane[index] == 0))
	//					ok = true;

					if(ok) {
						if(!destPlane[index])
							destPlane[index] = 1;
					}
					
				}
				i++;
				if(i >= width) {
					j++;
					i = 0;
				}
			}
		}
	}

	void volLocalStdDeviation(const CVolumeSet<T> &src, int radius, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		radius = std::max(0,radius);

		unsigned char finished = 0;
		unsigned char last_update = 0;

		#pragma omp parallel for
		for(int k = 0; k < depth; k++) {
			finished++;
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			if(thread_num == 0 && progress != NULL) {
				last_update = finished;
				progress->UpdateProgress((float)finished / (float)depth);
			}

			for(int j = 0; j < height; j++)
			for(int i = 0; i < width; i++) {
				int si, sj, sk;
				//compute mean
				double accum = 0;
				int64 voxels = 0;
				for(sk = std::max(k - radius, 0); sk <= std::min(k + radius, depth-1); sk++)
				for(sj = std::max(j - radius, 0); sj <= std::min(j + radius, height-1); sj++)
				for(si = std::max(i - radius, 0); si <= std::min(i + radius, width-1); si++) {
					accum += src.getValue(si,sj,sk);
					voxels++;
				}
				double mean = accum / (double)voxels;

				accum = 0;
				// compute sqrt(var)
				for(sk = std::max(k - radius, 0); sk <= std::min(k + radius, depth-1); sk++)
				for(sj = std::max(j - radius, 0); sj <= std::min(j + radius, height-1); sj++)
				for(si = std::max(i - radius, 0); si <= std::min(i + radius, width-1); si++) {
					double diff = mean - (double) src.getValue(si,sj,sk);
					accum += diff*diff;
				}
				
				setValue(i, j, k, (T)sqrt(accum/(double)voxels));
			}
		}
	}
	void volLocalVariance(const CVolumeSet<T> &src, int radius, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		radius = std::max(0,radius);

		unsigned char finished = 0;
		unsigned char last_update = 0;

		#pragma omp parallel for
		for(int k = 0; k < depth; k++) {
			finished++;
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			if(thread_num == 0 && progress != NULL) {
				last_update = finished;
				progress->UpdateProgress((float)finished / (float)depth);
			}

			for(int j = 0; j < height; j++)
			for(int i = 0; i < width; i++) {
				int si, sj, sk;
				// TODO: optimize, accept computed mean to prevent subsequent mean computation
				//compute mean 
				double accum = 0;
				int64 voxels = 0;
				for(sk = std::max(k - radius, 0); sk <= std::min(k + radius, depth-1); sk++)
				for(sj = std::max(j - radius, 0); sj <= std::min(j + radius, height-1); sj++)
				for(si = std::max(i - radius, 0); si <= std::min(i + radius, width-1); si++) {
					accum += src.getValue(si,sj,sk);
					voxels++;
				}
				double mean = accum / (double)voxels;

				accum = 0;
				// compute sqrt(var)
				for(sk = std::max(k - radius, 0); sk <= std::min(k + radius, depth-1); sk++)
				for(sj = std::max(j - radius, 0); sj <= std::min(j + radius, height-1); sj++)
				for(si = std::max(i - radius, 0); si <= std::min(i + radius, width-1); si++) {
					double diff = mean - (double) src.getValue(si,sj,sk);
					accum += diff*diff;
				}
				
				setValue(i, j, k, (T)(accum/(double)voxels));
			}
		}
	}

	void volLocalMean(const CVolumeSet<T> &src, int radius, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		radius = std::max(0,radius);

		unsigned char finished = 0;
		unsigned char last_update = 0;

		#pragma omp parallel for
		for(int k = 0; k < depth; k++) {
			finished++;
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			if(thread_num == 0 && progress != NULL) {
				last_update = finished;
				progress->UpdateProgress((float)finished / (float)depth);
			}

			for(int j = 0; j < height; j++)
			for(int i = 0; i < width; i++) {
				int si, sj, sk;
				//compute mean
				double accum = 0;
				int64 voxels = 0;
				for(sk = std::max(k - radius, 0); sk <= std::min(k + radius, depth-1); sk++)
				for(sj = std::max(j - radius, 0); sj <= std::min(j + radius, height-1); sj++)
				for(si = std::max(i - radius, 0); si <= std::min(i + radius, width-1); si++) {
					accum += src.getValue(si,sj,sk);
					voxels++;
				}
				double mean = accum / (double)voxels;

				setValue(i, j, k, (T)mean);
			}
		}
	}

	T getMean() const {
		int i, j;
		double accum = 0;
		int imgSize = width * height;
		for (i = 0; i < depth; i++) {
			for (j = 0; j < imgSize; j++)
				accum += (double) planes[i][j];
		}
		return (T) (accum / (width * height * depth));
	}

	T getStdDev() const {
		int i, j;
		double accum = 0;
		double mean = getMean();
		int imgSize = width * height;
		for (i = 0; i < depth; i++) {
			for (j = 0; j < imgSize; j++) {
				double diff = planes[i][j] - mean;
				accum += diff * diff;
			}
		}
		return (T) sqrt(accum / (width * height * depth));
	}

	T getVar() const {
		int i, j;
		double accum = 0;
		double mean = getMean();
		int imgSize = width * height;
		for (i = 0; i < depth; i++) {
			for (j = 0; j < imgSize; j++) {
				double diff = planes[i][j] - mean;
				accum += diff * diff;
			}
		}
		return (T) (accum / (width * height * depth));
	}

	std::vector<T> getNeighborhood(SPoint3D<int> pt, unsigned int radius) const {
		// new version, we need to have voxels corresponding to the center of the volume to be in the middle of the vector
		std::vector<T> retval;
		int i, j, k;
		for(k = pt.z - (int)radius; k <= pt.z + (int)radius; k++) 
		for(j = pt.y - (int)radius; j <= pt.y + (int)radius; j++)
		for(i = pt.x - (int)radius; i <= pt.x + (int)radius; i++) {
			if(k >= 0 && k < depth && j >= 0 && j < height && i >= 0 && i < width)
				retval.push_back(getValue(i, j, k));
			else
				retval.push_back(0);
		}

		return retval;
	}

	void getNeighborhoodNoResize(std::vector<T> &dstVector, SPoint3D<int> pt, unsigned int radius) const {
		// always fills the vector with all values, nondefined values filled as 0
		int i, j, k;
		int idx = 0;
		for(k = pt.z - (int)radius; k <= pt.z + (int)radius; k++) 
		for(j = pt.y - (int)radius; j <= pt.y + (int)radius; j++)
		for(i = pt.x - (int)radius; i <= pt.x + (int)radius; i++) {
			if(k >= 0 && k < depth && j >= 0 && j < height && i >= 0 && i < width)
				dstVector[idx] = (getValue(i, j, k));
			else
				dstVector[idx] = (0);
			idx++;
		}
	}

	void getNeighborhoodFast(T *dstVector, SPoint3D<int> pt, unsigned int radius) const {
		// fast version, no allocs, dstVector must be of size (2*radius+1)^3
		int i, j, k;
		int idx = 0;
		for(k = pt.z - (int)radius; k <= pt.z + (int)radius; k++) 
		for(j = pt.y - (int)radius; j <= pt.y + (int)radius; j++)
		for(i = pt.x - (int)radius; i <= pt.x + (int)radius; i++) {
			if(k >= 0 && k < depth && j >= 0 && j < height && i >= 0 && i < width)
				dstVector[idx++] = getValue(i, j, k);
			else
				dstVector[idx++] = 0;
		}
	}

	T getSum() const {
		T accum = 0;
		int i, j;
		int imgSize = width * height;
		for (i = 0; i < depth; i++) {
			for (j = 0; j < imgSize; j++)
				accum += planes[i][j];
		}
		return accum;
	}

	void getIntApproxHistogram(std::vector<int> &histo, int &minValue, int &maxValue) const {
		int i, j;
		int imgSize = width * height;
		T min, max;
		min = max = planes[0][0];
		for (i = 0; i < depth; i++) {
			for (j = 0; j < imgSize; j++) {
				T val = planes[i][j];
				if(val < min)
					min = val;
				if(val > max)
					max = val;
			}
		}
		int iMin, iMax;
		iMin = (int) min - 1;
		iMax = (int) max + 1;
		int iVals = (iMax - iMin);
		histo.resize(iVals);

		for (i = 0; i < depth; i++) {
			for (j = 0; j < imgSize; j++) {
				T val = planes[i][j];
				int convVal = (int) (val - min + .5);
				histo[std::max(0, std::min(iVals - 1, convVal))]++;
			}
		}

		minValue = iMin;
		maxValue = iMax;
	}

	// performs a NL means filter on src, computed on +-radius and local neighborhoods are size +-neighborhood
	void volBasicNLMeans(const CVolumeSet<T> &src, T h, int radius, int neighborhood, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		radius = std::max(0,radius);

		unsigned int finished = 0;
		unsigned int last_update = 0; 

		progress->UpdateProgress(0);

		#pragma omp parallel for //num_threads(8)
		for(int k = 0; k < depth; k++) {
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			int workSize = 2*radius + 1;
			CVolumeSet<double> workVol(workSize, workSize, workSize);

			int nbhsize = (2*neighborhood + 1);
			nbhsize = nbhsize * nbhsize * nbhsize;
			T* targetNbh = new T[nbhsize];
			T* tempNbh = new T[nbhsize];

			for(int j = 0; j < height; j++) {

				// update progress
				if(thread_num == 0 && progress != NULL) {
					last_update = finished;
					progress->UpdateProgress((float)finished / ((float)depth*(float)height));
				}

				finished++;

				for(int i = 0; i < width; i++) {
					SPoint3D<int> workPos(i-radius, j-radius, k-radius);

					src.getNeighborhoodFast(targetNbh, SPoint3D<int>(i,j,k), neighborhood);

					int si, sj, sk;
					int skmin = std::max(0,k - radius), skmax = std::min(depth-1,k+radius);
					int sjmin = std::max(0,j - radius), sjmax = std::min(height-1,j+radius);
					int simin = std::max(0,i - radius), simax = std::min(width-1,i+radius);

					// compute weight for each voxel in the radius
					for(sk = skmin; sk <= skmax; sk++)
					for(sj = sjmin; sj <= sjmax; sj++)
					for(si = simin; si <= simax; si++) {
						src.getNeighborhoodFast(tempNbh, SPoint3D<int>(si, sj, sk), neighborhood);
						workVol.setValue(si - simin, sj - sjmin, sk - skmin, 
							exp(-(double) L2NormSq(targetNbh, tempNbh, nbhsize) / ((double)h*(double)h)));
					}
					// normalize weights
					double sum = workVol.getSum();
					workVol.multiply(1./sum);

					double result = 0;
					// compute result by multiplying original data with computed weights
					for(sk = skmin; sk <= skmax; sk++)
					for(sj = sjmin; sj <= sjmax; sj++)
					for(si = simin; si <= simax; si++) {
						result += (double) src.getValue(si, sj, sk) * workVol.getValue(si - simin, sj - sjmin, sk - skmin);
					}
					setValue(i, j, k, (T)result);
				}
			}

			delete[] targetNbh;
			delete[] tempNbh;
		}
	}

	void volPseudoResiduals(const CVolumeSet<T> &src, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		unsigned int finished = 0;
		unsigned int last_update = 0;

		progress->UpdateProgress(0);
		double sqrt6div7 = sqrt(6./7.);

		#pragma omp parallel for
		for(int k = 0; k < depth; k++) {
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			for(int j = 0; j < height; j++) {

				// update progress
				if(thread_num == 0 && progress != NULL) {
					last_update = finished;
					progress->UpdateProgress((float)finished / ((float)depth*(float)height));
				}

				finished++;

				for(int i = 0; i < width; i++) {
					// get values from 6-neighborhood
					double accum = 0;
					if(i > 0)
						accum += src.getValue(i-1, j, k);
					if(i < width - 1)
						accum += src.getValue(i+1, j, k);
					if(j > 0)
						accum += src.getValue(i, j-1, k);
					if(j < height - 1)
						accum += src.getValue(i, j+1, k);
					if(k > 0)
						accum += src.getValue(i, j, k-1);
					if(k < depth - 1)
						accum += src.getValue(i, j, k+1);

					// compute pseudo-residual value 
					double res = sqrt6div7 * (src.getValue(i,j,k) - (1./6.) * accum);

					setValue(i, j, k, (T)res);
				}
			}
		}
	}

	T getVarianceFromPseudoRes() {
		int i, j;
		double accum = 0;
		int imgSize = width * height;
		for (i = 0; i < depth; i++) {
			for (j = 0; j < imgSize; j++)
			{
				double val = (double) planes[i][j];
				accum += val*val;
			}
		}
		return (T) (accum / (width * height * depth));
	}

	// performs a NL means filter on src, computed on +-radius and local neighborhoods are size +-neighborhood
	void volNLMeans(const CVolumeSet<T> &src, T beta, int radius, int neighborhood, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		int nbhsize = (2*neighborhood + 1);
		nbhsize = nbhsize * nbhsize * nbhsize;
			
		CVolumeSet<T> pseudores(0,0,0);
		pseudores.volPseudoResiduals(src, progress);
		double variance = (double) pseudores.getVarianceFromPseudoRes();
		double weightConst = 2 * beta * variance * nbhsize;

		radius = std::max(0,radius);

		unsigned int finished = 0;
		unsigned int last_update = 0; 

		progress->UpdateProgress(0);

		#pragma omp parallel for schedule(dynamic)//num_threads(1)
		for(int k = 0; k < depth; k++) {
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			int workSize = 2*radius + 1;
			CVolumeSet<double> workVol(workSize, workSize, workSize);

			T* targetNbh = new T[nbhsize];
			T* tempNbh = new T[nbhsize];

			for(int j = 0; j < height; j++) {

				// update progress
				if(thread_num == 0 && progress != NULL) {
					last_update = finished;
					progress->UpdateProgress((float)finished / ((float)depth*(float)height));
				}

				finished++;

				for(int i = 0; i < width; i++) {
					SPoint3D<int> workPos(i-radius, j-radius, k-radius);

					src.getNeighborhoodFast(targetNbh, SPoint3D<int>(i,j,k), neighborhood);

					int si, sj, sk;
					int skmin = std::max(0,k - radius), skmax = std::min(depth-1,k+radius);
					int sjmin = std::max(0,j - radius), sjmax = std::min(height-1,j+radius);
					int simin = std::max(0,i - radius), simax = std::min(width-1,i+radius);

					// compute weight for each voxel in the radius
					for(sk = skmin; sk <= skmax; sk++)
					for(sj = sjmin; sj <= sjmax; sj++)
					for(si = simin; si <= simax; si++) {
						src.getNeighborhoodFast(tempNbh, SPoint3D<int>(si, sj, sk), neighborhood);
						workVol.setValue(si - simin, sj - sjmin, sk - skmin, 
							exp(-(double) L2NormSq(targetNbh, tempNbh, nbhsize) / (weightConst)));
					}
					// normalize weights
					double sum = workVol.getSum();
//					workVol.multiply(1./sum);

					double result = 0;
					// compute result by multiplying original data with computed weights
					for(sk = skmin; sk <= skmax; sk++)
					for(sj = sjmin; sj <= sjmax; sj++)
					for(si = simin; si <= simax; si++) {
						result += (double) src.getValue(si, sj, sk) * workVol.getValue(si - simin, sj - sjmin, sk - skmin);
					}
					setValue(i, j, k, (T)(result / sum));
				}
			}

			delete[] targetNbh;
			delete[] tempNbh;
		}
	}

	void volOptimizedNLMeans(const CVolumeSet<T> &src, double beta, double meanAccept, double sigma1, int radius, int neighborhood, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		int nbhsize = (2*neighborhood + 1);
		nbhsize = nbhsize * nbhsize * nbhsize;
			
		CVolumeSet<T> pseudores(0,0,0);
		pseudores.volPseudoResiduals(src, progress);
		double variance = (double) pseudores.getVarianceFromPseudoRes();
		double weightConst = 2 * beta * variance * nbhsize;

		CVolumeSet<T> means(1,1,1), variances(1,1,1);
		means.volLocalMean(src, neighborhood, progress);
		variances.volLocalVariance(src, neighborhood, progress);

		radius = std::max(0,radius);

		unsigned int finished = 0;
		unsigned int last_update = 0; 

		progress->UpdateProgress(0);

		#pragma omp parallel for //num_threads(8)
		for(int k = 0; k < depth; k++) {
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			int workSize = 2*radius + 1;
			double *workValues = new double[workSize * workSize * workSize];
			T *srcValues = new T[workSize * workSize * workSize];

			T* targetNbh = new T[nbhsize];
			T* tempNbh = new T[nbhsize];

			for(int j = 0; j < height; j++) {

				// update progress
				if(thread_num == 0 && progress != NULL) {
					last_update = finished;
					progress->UpdateProgress((float)finished / ((float)depth*(float)height));
				}

				finished++;

				for(int i = 0; i < width; i++) {
					SPoint3D<int> workPos(i-radius, j-radius, k-radius);

					src.getNeighborhoodFast(targetNbh, SPoint3D<int>(i,j,k), neighborhood);
					T mean1 = means.getValue(i,j,k);
					T var1 = variances.getValue(i,j,k);

					int si, sj, sk;
					int skmin = std::max(0,k - radius), skmax = std::min(depth-1,k+radius);
					int sjmin = std::max(0,j - radius), sjmax = std::min(height-1,j+radius);
					int simin = std::max(0,i - radius), simax = std::min(width-1,i+radius);

					int computedValues = 0;
					// compute weight for each voxel in the radius
					for(sk = skmin; sk <= skmax; sk++)
					for(sj = sjmin; sj <= sjmax; sj++)
					for(si = simin; si <= simax; si++) {
						T mean2 = means.getValue(si,sj,sk);
						T var2 = variances.getValue(si,sj,sk);
						double var1divvar2 = var1 / var2;

						if((fabs(mean1 - mean2) <= variance * meanAccept) &&
							(sigma1 < var1divvar2 && var1divvar2 < 1./sigma1)){
							src.getNeighborhoodFast(tempNbh, SPoint3D<int>(si, sj, sk), neighborhood);
							workValues[computedValues] = exp(-(double) L2NormSq(targetNbh, tempNbh, nbhsize) / (weightConst));
							srcValues[computedValues] = src.getValue(si, sj, sk);
							computedValues++;
						}

					}
					// normalize weights
					double sum = 0;
					for(si = 0; si < computedValues; si++)
						sum += workValues[si];
//					workVol.multiply(1./sum);

					double result = 0;
					// compute result by multiplying original data with computed weights
					for(si = 0; si < computedValues; si++) {
						result += (double) srcValues[si] * workValues[si];
					}
					setValue(i, j, k, (T)(result / sum));
				}
			}

			delete[] workValues;
			delete[] targetNbh;

			delete[] tempNbh;
		}
	}

	void volBlockwiseNLMeans(const CVolumeSet<T> &src, double beta, double meanAccept, double sigma1, int radius, int blockNbh, int blockDist, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		// neighborhood size
		int nbhsize = (2*blockNbh + 1);
		nbhsize = nbhsize * nbhsize * nbhsize;
			
		// compute automatic weight constants and total image variance
		CVolumeSet<T> pseudores(0,0,0);
		pseudores.volPseudoResiduals(src, progress);
		double variance = (double) pseudores.getVarianceFromPseudoRes();
		double weightConst = 2 * beta * variance * nbhsize;

		CVolumeSet<T> means(1,1,1), variances(1,1,1);
		means.volLocalMean(src, blockNbh, progress);
		variances.volLocalVariance(src, blockNbh, progress);

		CVolumeSet<T> accum(1,1,1); // here will be accumulated results from the blocks
		accum.copySize(src);
		CVolumeSet<unsigned int> numAccum(1,1,1); // number of values stored in each position from each block
		numAccum.copySize(src);
	
		accum.setZero();
		numAccum.setZero();

		radius = std::max(0,radius);

		unsigned int finished = 0;
		unsigned int last_update = 0; 
		//exp_fast_prepare();

		progress->UpdateProgress(0);

		#pragma omp parallel for schedule(dynamic)
		for(int k = blockNbh; k < depth; k += blockDist) {
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			int workSize = 2*radius + 1;
			double *workValues = new double[workSize * workSize * workSize];
			SPoint3D<int> *workValCoords = new SPoint3D<int>[workSize * workSize * workSize];

			T* targetNbh = new T[nbhsize];
			T* tempNbh = new T[nbhsize];

			for(int j = blockNbh; j < height; j += blockDist) {


				// update progress
				if(thread_num == 0 && progress != NULL) {
					last_update = finished;
					progress->UpdateProgress((float)finished / ((float)(depth / blockDist)*(float)(height / blockDist)));
				}

				finished++;

				for(int i = blockNbh; i < width; i += blockDist) {
					SPoint3D<int> workPos(i-radius, j-radius, k-radius);

					src.getNeighborhoodFast(targetNbh, SPoint3D<int>(i,j,k), blockNbh);
					T mean1 = means.getValue(i,j,k);
					T var1 = variances.getValue(i,j,k);

					int si, sj, sk;
					int skmin = std::max(0,k - radius), skmax = std::min(depth-1,k+radius);
					int sjmin = std::max(0,j - radius), sjmax = std::min(height-1,j+radius);
					int simin = std::max(0,i - radius), simax = std::min(width-1,i+radius);

					int computedValues = 0;
					// compute weight for each voxel in the radius
					for(sk = skmin; sk <= skmax; sk++)
					for(sj = sjmin; sj <= sjmax; sj++)
					for(si = simin; si <= simax; si++) {
						T mean2 = means.getValue(si,sj,sk);
						T var2 = variances.getValue(si,sj,sk);
						double var1divvar2 = var1 / var2;

						if((fabs(mean1 - mean2) <= variance * meanAccept) &&
							(sigma1 < var1divvar2 && var1divvar2 < 1./sigma1)){
							src.getNeighborhoodFast(tempNbh, SPoint3D<int>(si, sj, sk), blockNbh);
							workValues[computedValues] = exp(-(double) L2NormSq(targetNbh, tempNbh, nbhsize) / (weightConst));
							workValCoords[computedValues].x = si;
							workValCoords[computedValues].y = sj;
							workValCoords[computedValues].z = sk;
							computedValues++;
						}

					}
					// normalize weights
					double sum = 0;
					for(si = 0; si < computedValues; si++)
						sum += workValues[si];
					for(si = 0; si < computedValues; si++)
						workValues[si] /= sum;

					skmin = std::max(0, k - blockNbh); skmax = std::min(depth-1, k + blockNbh);
					sjmin = std::max(0, j - blockNbh); sjmax = std::min(height-1, j + blockNbh);
					simin = std::max(0, i - blockNbh); simax = std::min(width-1, i + blockNbh);
					
					skmin -= k; skmax -= k;
					sjmin -= j; sjmax -= j;
					simin -= i; simax -= i;

					// for all relevant blocks -  copy them to the result
					for(int val = 0; val < computedValues; val++) {
						double blockWeight = workValues[val];
						SPoint3D<int> &blockOrigin = workValCoords[val];

						for(sk = skmin; sk <= skmax; sk++)
						for(sj = sjmin; sj <= sjmax; sj++)
						for(si = simin; si <= simax; si++) {

							if(blockOrigin.x + si >= 0 && blockOrigin.x + si < width && 
								blockOrigin.y + sj >= 0 && blockOrigin.y + sj < height && 
								blockOrigin.z + sk >= 0 && blockOrigin.z + sk < depth)
							{
								double result = 0;
								result = src.getValue(blockOrigin.x + si, blockOrigin.y + sj, blockOrigin.z + sk) * blockWeight;
								int offset = (sj + j) * width + (si + i);
								accum.planes[sk+k][offset] += (T)result;
							}
						}
					}

					for(sk = skmin; sk <= skmax; sk++)
					for(sj = sjmin; sj <= sjmax; sj++)
					for(si = simin; si <= simax; si++) {
						int offset = (sj + j) * width + (si + i);
						numAccum.planes[sk+k][offset]++;
					}

					//setValue(i, j, k, (T)(accum.getValue(i,j,k) / (T)numAccum.getValue(i,j,k)));
				}
			}

			delete[] workValues;
			delete[] targetNbh;

			delete[] tempNbh;
		}

		int total = width * height;
		for(int k = 0; k < depth; k++) {
			for(int i = 0; i < total; i++) {
				planes[k][i] = accum.planes[k][i] / (T)numAccum.planes[k][i];
			}	
		}
	}

	void getSubVolume(const CVolumeSet<T> &src, SPoint3D<int> &orig, SPoint3D<int> &size) {
		realloc(size.x, size.y, size.z);
		origPos = orig;

		for(int z = 0; z < depth; z++) {
		for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			T val;
			int sx, sy, sz;
			sx = x + orig.x;
			sy = y + orig.y;
			sz = z + orig.z;
			if(sx < 0 || sx >= src.width || sy < 0 || sy >= src.height || sz < 0 || sz >= src.depth)
				val = 0;
			else
				val = src.getValue(sx, sy, sz);
			setValue(x, y, z, val);
		}
		}
		}
	}

	void setSubvolume(const CVolumeSet<T> &subVol, int ignoreBorder) {
		SPoint3D<int> orig = subVol.origPos;

		for(int z = ignoreBorder; z < subVol.depth - ignoreBorder; z++) {
		for(int y = ignoreBorder; y < subVol.height - ignoreBorder; y++) {
		for(int x = ignoreBorder; x < subVol.width - ignoreBorder; x++) {
			int dx, dy, dz;
			dx = x + orig.x;
			dy = y + orig.y;
			dz = z + orig.z;
			if(!(dx < 0 || dx >= width || dy < 0 || dy >= height || dz < 0 || dz >= depth)) {
				T val;
				val = subVol.getValue(x, y, z);
				setValue(dx, dy, dz, val);
			} 
		}
		}
		}
	}

	// saves subvolume directly to an array
	void saveSubvolumeToArray(T *pDest, SPoint3D<int> &orig, SPoint3D<int> &size) const {
		int offset = 0;
		for(int z = 0; z < size.z; z++) {
			for(int y = 0; y < size.y; y++) {
				for(int x = 0; x < size.x; x++) {
					T val;
					int sx, sy, sz;
					sx = x + orig.x;
					sy = y + orig.y;
					sz = z + orig.z;
					if(sx < 0 || sx >= width || sy < 0 || sy >= height || sz < 0 || sz >= depth)
						val = 0;
					else
						val = getValue(sx, sy, sz);
					pDest[offset++] = val; 
				}
			}
		}
	}
	
	// loads subvolume directly from an array
	void loadSubvolumeFromArray(const T *pSrc, SPoint3D<int> &orig, SPoint3D<int> &size, int ignoreBorder = 0) {
		int offset = 0;
		for(int z = 0; z < size.z ; z++) {
			for(int y = 0; y < size.y; y++) {
				for(int x = 0; x < size.x; x++) {
					int sx, sy, sz;
					sx = x + orig.x;
					sy = y + orig.y;
					sz = z + orig.z;
					if(false == (sx < 0 || sx >= width || sy < 0 || sy >= height || sz < 0 || sz >= depth)) {
						if(false == (x < ignoreBorder || x >= size.x - ignoreBorder || 
						   y < ignoreBorder || y >= size.y - ignoreBorder || 
						   z < ignoreBorder || z >= size.z - ignoreBorder))
							setValue(sx, sy, sz, pSrc[offset]);
					}
					offset++;
				}
			}
		}
	}
	
	// saves volumeset into array, MUST be allocated to the width*height*depth*sizeof(T) size
	void saveToArray(T *pDest) {
		int imgSize = width * height;
		int destOffset = 0;
		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < imgSize; j++)
				pDest[destOffset++] = planes[i][j];
		}	
	}

	// loads volumeset from array, MUST be allocated to the width*height*depth*sizeof(T) size
	void loadFromArray(const T *pSrc) {
		int imgSize = width * height;
		int srcOffset = 0;
		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < imgSize; j++)
				planes[i][j] = pSrc[srcOffset++];
		}	
	}

	// performs a blockwise NL means filter on src, subdivides data if necessary
	void volBlockwiseNLMeansBig(const CVolumeSet<T> &src, const SPoint3D<int> &maxComputeSize, double beta, double meanAccept, double sigma1, int radius, int blockNbh, int blockDist, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);
//		const int maxWidth = 150, maxHeight = 150, maxDepth = 150;
		std::vector<int> vStartX, vStartY, vStartZ;
		std::vector<int> vWidthX, vWidthY, vWidthZ;
		
		int border = 1 + radius + blockNbh;

		int offset;
		offset = 0;
		while(offset < src.width) {
			vStartX.push_back(offset);
			vWidthX.push_back(std::min(maxComputeSize.x, src.width - offset));
			offset += maxComputeSize.x;
		}	
		offset = 0;
		while(offset < src.height) {
			vStartY.push_back(offset);
			vWidthY.push_back(std::min(maxComputeSize.y, src.height - offset));
			offset += maxComputeSize.y;
		}	
		offset = 0;
		while(offset < src.depth) {
			vStartZ.push_back(offset);
			vWidthZ.push_back(std::min(maxComputeSize.z, src.depth - offset));
			offset += maxComputeSize.z;
		}	

		for(int k = 0; k < (int)vStartZ.size(); k++) 
		for(int j = 0; j < (int)vStartY.size(); j++) 
		for(int i = 0; i < (int)vStartX.size(); i++) {
			SPoint3D<int> orig, size;
			orig.x = vStartX[i] - border;
			orig.y = vStartY[j] - border;
			orig.z = vStartZ[k] - border;
			size.x = vWidthX[i] + 2*border;
			size.y = vWidthY[j] + 2*border;
			size.z = vWidthZ[k] + 2*border;

			CVolumeSet<T> subVol(1,1,1), workVol(1,1,1);
			subVol.getSubVolume(src, orig, size);
			workVol.volBlockwiseNLMeans(subVol, beta, meanAccept, sigma1, radius, blockNbh, blockDist, progress);
			setSubvolume(workVol, border);
		}
	}

	// performs a blockwise NL means filter on src, subdivides data if necessary
	void volNLMeansBig(const CVolumeSet<T> &src, const SPoint3D<int> &maxComputeSize, double beta, int radius, int blockNbh, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);
//		const int maxWidth = 150, maxHeight = 150, maxDepth = 150;
		std::vector<int> vStartX, vStartY, vStartZ;
		std::vector<int> vWidthX, vWidthY, vWidthZ;
		
		int border = 1 + radius + blockNbh;

		int offset;
		offset = 0;
		while(offset < src.width) {
			vStartX.push_back(offset);
			vWidthX.push_back(std::min(maxComputeSize.x, src.width - offset));
			offset += maxComputeSize.x;
		}	
		offset = 0;
		while(offset < src.height) {
			vStartY.push_back(offset);
			vWidthY.push_back(std::min(maxComputeSize.y, src.height - offset));
			offset += maxComputeSize.y;
		}	
		offset = 0;
		while(offset < src.depth) {
			vStartZ.push_back(offset);
			vWidthZ.push_back(std::min(maxComputeSize.z, src.depth - offset));
			offset += maxComputeSize.z;
		}	

		for(int k = 0; k < (int)vStartZ.size(); k++) 
		for(int j = 0; j < (int)vStartY.size(); j++) 
		for(int i = 0; i < (int)vStartX.size(); i++) {
			SPoint3D<int> orig, size;
			orig.x = vStartX[i] - border;
			orig.y = vStartY[j] - border;
			orig.z = vStartZ[k] - border;
			size.x = vWidthX[i] + 2*border;
			size.y = vWidthY[j] + 2*border;
			size.z = vWidthZ[k] + 2*border;

			CVolumeSet<T> subVol(1,1,1), workVol(1,1,1);
			subVol.getSubVolume(src, orig, size);
			workVol.volNLMeans(subVol, (float)beta, radius, blockNbh, progress);
			setSubvolume(workVol, border);
		}
	}

	void volMedianFilter(const CVolumeSet<T> &src, int radius, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		// neighborhood size
		int nbhsize = (2*radius + 1);
		nbhsize = nbhsize * nbhsize * nbhsize;

		unsigned int finished = 0;
		unsigned int last_update = 0; 

		if(progress)
			progress->UpdateProgress(0);

		#pragma omp parallel for //num_threads(8)
		for(int k = 0; k < depth; k ++) {
			unsigned int thread_num, thread_total;
			thread_num = omp_get_thread_num();
			thread_total = omp_get_num_threads();
			
			std::vector<T> tempNbh;
			tempNbh.resize(nbhsize);

			for(int j = 0; j < height; j ++) {

				// update progress
				if(thread_num == 0 && progress != NULL) {
					last_update = finished;
					if(progress)
						progress->UpdateProgress((float)finished / ((float)(depth)*(float)(height)));
				}

				finished++;

				for(int i = 0; i < width; i ++) {
					src.getNeighborhoodNoResize(tempNbh, SPoint3D<int>(i,j,k), radius);
					std::nth_element(tempNbh.begin(), tempNbh.begin() + (tempNbh.size()/2), tempNbh.end());
					setValue(i, j, k, tempNbh[tempNbh.size()/2]);
				}
			}

		}
	}

#ifdef OPENCL
	bool volNLMeansHW(MyOpenCL &ocl, const CVolumeSet<T> &src,
							   double beta, int radius, int neighbourhood,
							   CProgress *progress) {
		if(ocl.bInitialized == false) {
			printf("OpenCL not initialized\n");
			return false;
		}
		
		if(radius > 4) {
			radius = 4;
			printf("Radius too large, setting to 4.\n");
		}
		if(neighbourhood > 2) {
			radius = 2;
			printf("Neighbourhood too large, setting to 2.\n");
		}

		// create program
		if(NULL == ocl.ReadNLMProgram("NLMProgram.cl")) {
			printf("Cannot load program\n");
			return false;
		}
		cl_program program = clCreateProgramWithSource(ocl.context, 1, (const char**) &ocl.szNLMProgram, NULL, NULL);
		if(program == NULL) {
			printf("Cannot create program\n");
			return false;
		}

		// bulid program
		int errcode = clBuildProgram(program, 1, &(ocl.device), NULL, NULL, NULL);
		if (errcode != CL_SUCCESS) {
			size_t len;
			char buffer[10000];
			printf("Error: Failed to build program executable (%d)\n", errcode);            
			clGetProgramBuildInfo(program, ocl.device, CL_PROGRAM_BUILD_LOG,
											  sizeof(buffer), buffer, &len);
			printf("%s\n", buffer);
			return false;
		}

		// create kenerl
		cl_kernel kerNLM = clCreateKernel(program, "NLMeansV2", NULL);
		cl_kernel kerPseudoRes = clCreateKernel(program, "PseudoRes", NULL);
//		cl_kernel kerMeanVar = clCreateKernel(program, "LocMeanAndVar", NULL);

		copySize(src);
		copyOrigPos(src);

		std::vector<int> vStartX, vStartY, vStartZ;
		std::vector<int> vWidthX, vWidthY, vWidthZ;
		
		int border = radius + neighbourhood;

		const int BLOCK_NLOPT=13;

		const int BLOCK_WEIGHTS=9;

		int xkern = BLOCK_NLOPT - 2*border; // size of "kernel", i.e. voxels that are actually computed in X direction
		int ykern = BLOCK_NLOPT - 2*border; // size of "kernel", i.e. voxels that are actually computed in Y direction

		SPoint3D<int> blockSize, blocks;
		blocks.init(std::min(100, (width + xkern - 1)/xkern), std::min(100, (height + ykern - 1)/ykern), depth);
		blockSize.init(blocks.x * xkern, blocks.y * ykern, std::min(100, depth));

		int offset;
		offset = 0;
		while(offset < src.width) {
			vStartX.push_back(offset);
			vWidthX.push_back(std::min(blockSize.x, src.width - offset));
			offset += blockSize.x;
		}	
		offset = 0;
		while(offset < src.height) {
			vStartY.push_back(offset);
			vWidthY.push_back(std::min(blockSize.y, src.height - offset));
			offset += blockSize.y;
		}	
		offset = 0;
		while(offset < src.depth) {
			vStartZ.push_back(offset);
			vWidthZ.push_back(std::min(blockSize.z, src.depth - offset));
			offset += blockSize.z;
		}	

		Timer timer;
		timer.restart();

		printf("Starting computation\n");
		printf("Preprocessing pseudoresiduals\n");

		int blocksTotal = (int)(vStartX.size() * vStartY.size() * vStartZ.size());
		int blocksDone = 0;
		double accum = 0;
		int accumVals = 0;
		for(int k = 0; k < (int)vStartZ.size(); k++) 
		for(int j = 0; j < (int)vStartY.size(); j++) 
		for(int i = 0; i < (int)vStartX.size(); i++) {
			printf("Computing Pseudoresiduals block %d/%d\n", ++blocksDone, blocksTotal);
			
			int sizeAsArray[4]; // size must be 4-component, because we use it as parameter for kernels
			SPoint3D<int> &size = *((SPoint3D<int>*)sizeAsArray);
			SPoint3D<int> orig;
			int pseudoresBorder = 1;
			orig.x = vStartX[i] - pseudoresBorder;
			orig.y = vStartY[j] - pseudoresBorder;
			orig.z = vStartZ[k] - pseudoresBorder;
			SPoint3D<int> gpuBlocks;
			gpuBlocks.init(vWidthX[i], vWidthY[j], vWidthZ[k]);
			size.x = gpuBlocks.x + 2*pseudoresBorder;
			size.y = gpuBlocks.y + 2*pseudoresBorder;
			size.z = gpuBlocks.z + 2*pseudoresBorder;

			int numEntries = size.x * size.y * size.z;
			int numPseudores = gpuBlocks.x * gpuBlocks.y * gpuBlocks.z;

			size_t iGlobSize = gpuBlocks.y * gpuBlocks.x; // one kernel for each line in X direction
			size_t iLocSize = gpuBlocks.x;

			float *pVolArray = new float[numEntries];
			float *pPseudoresArray = new float[numPseudores];

			memset(pPseudoresArray, 0, sizeof(float)*numPseudores);
			src.saveSubvolumeToArray(pVolArray, orig, size);

			// alloc memory
			cl_mem memSrc, memPseudores;
			memSrc = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*numEntries, pVolArray, NULL);
			memPseudores = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)*numPseudores, NULL, NULL);
			
			//set arguments
			clSetKernelArg(kerPseudoRes, 0, sizeof(cl_mem), (void*)&memSrc);
			clSetKernelArg(kerPseudoRes, 1, sizeof(cl_mem), (void*)&memPseudores);
			clSetKernelArg(kerPseudoRes, 2, sizeof(cl_int4), &size.x);
			
			// TODO: make pseudores faster
			errcode = clEnqueueNDRangeKernel(ocl.queue, kerPseudoRes, 1, NULL, &iGlobSize, NULL, 0, NULL, NULL);
			clEnqueueReadBuffer(ocl.queue, memPseudores, CL_TRUE, 0, sizeof(float) * numPseudores, pPseudoresArray, 0, NULL, NULL);
			clReleaseMemObject(memPseudores);
			clReleaseMemObject(memSrc);

			// compute variance from pseudoresidual
			int idx = 0;
			for(idx = 0; idx < numPseudores; idx++) {
				double val = (double) pPseudoresArray[idx];
				accum += val*val;
				accumVals++;
			}

			//loadSubvolumeFromArray(pPseudoresArray, SPoint3D<int>(vStartX[i],vStartY[j],vStartZ[k]), gpuBlocks, 0);
			delete[] pPseudoresArray;
			delete[] pVolArray;
		}

		float variance = (float)((double)accum / (double)(accumVals));
		int nbhsize = (neighbourhood*2)+1;
		nbhsize = nbhsize * nbhsize * nbhsize;
		float weightConst = (float) (2 * (float)beta * variance * (float)(nbhsize));

		blocksDone = 0;
		for(int k = 0; k < (int)vStartZ.size(); k++) 
		for(int j = 0; j < (int)vStartY.size(); j++) 
		for(int i = 0; i < (int)vStartX.size(); i++) {
			printf("Computing block %d/%d\n", ++blocksDone, blocksTotal);
			
			int sizeAsArray[4]; // size must be 4-component, because we use it as parameter for kernels
			SPoint3D<int> &size = *((SPoint3D<int>*)sizeAsArray);
			SPoint3D<int> orig;
			orig.x = vStartX[i] - border;
			orig.y = vStartY[j] - border;
			orig.z = vStartZ[k] - border;
			SPoint3D<int> gpuBlocks;
			gpuBlocks.init(vWidthX[i], vWidthY[j], 1);
			size.x = gpuBlocks.x + 2*border;
			size.y = gpuBlocks.y + 2*border;
			size.z = vWidthZ[k] + 2*border;

			int numEntries = size.x * size.y * size.z;

			float *pVolArray = new float[numEntries];

			memset(pVolArray, 0, sizeof(float)*numEntries);
			src.saveSubvolumeToArray(pVolArray, orig, size);

			cl_mem memSrc = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*numEntries, pVolArray, NULL);
			cl_mem memDest = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)*numEntries, NULL, NULL);
			
			printf("Computing NM optimized filter...\n");
			//set arguments
			clSetKernelArg(kerNLM, 0, sizeof(cl_mem), (void*)&memSrc);
			clSetKernelArg(kerNLM, 1, sizeof(cl_mem), (void*)&memDest);
			clSetKernelArg(kerNLM, 2, sizeof(cl_float), &weightConst);
			clSetKernelArg(kerNLM, 3, sizeof(cl_int), &radius);
			clSetKernelArg(kerNLM, 4, sizeof(cl_int), &neighbourhood);
			clSetKernelArg(kerNLM, 5, sizeof(cl_int4), &size.x);
			clSetKernelArg(kerNLM, 6, sizeof(cl_float) * BLOCK_NLOPT * BLOCK_NLOPT * BLOCK_NLOPT, NULL);
			clSetKernelArg(kerNLM, 7, sizeof(cl_float) * BLOCK_WEIGHTS * BLOCK_WEIGHTS * BLOCK_WEIGHTS, NULL);
			clSetKernelArg(kerNLM, 8, sizeof(cl_float) * BLOCK_WEIGHTS * BLOCK_WEIGHTS, NULL);
			clSetKernelArg(kerNLM, 9, sizeof(cl_float) * BLOCK_WEIGHTS, NULL);
					
			size_t globs[3], locs[3];
			locs[0] = BLOCK_NLOPT; 
			locs[1] = BLOCK_NLOPT;
			locs[2] = BLOCK_NLOPT;
			globs[0] = locs[0] * gpuBlocks.x;
			globs[1] = locs[1] * gpuBlocks.y;
			globs[2] = locs[2] * gpuBlocks.z;

			errcode = clEnqueueNDRangeKernel(ocl.queue, kerNLM, 2, NULL, globs, locs, 0, NULL, NULL);
			memset(pVolArray, 0, sizeof(float) * numEntries);
			clEnqueueReadBuffer(ocl.queue, memDest, CL_TRUE, 0, sizeof(float) * numEntries, pVolArray, 0, NULL, NULL);
			
			printf("Storing results...\n");
			loadSubvolumeFromArray(pVolArray, orig, size, border);

			delete[] pVolArray;

			clReleaseMemObject(memDest);
			clReleaseMemObject(memSrc);
		}
		
		timer.measure();
		int h,m,s,ms;
		timer.getTime(h, m, s, ms);
		printf("Time taken: %d:%d:%d.%3d\n", h,m,s,ms);
		return true;
	}

	bool volNLMeansOptHW(MyOpenCL &ocl, const CVolumeSet<T> &src,
							   double beta, int radius, int neighbourhood,
							   CProgress *progress) {
		if(ocl.bInitialized == false) {
			printf("OpenCL not initialized\n");
			return false;
		}
		
		if(radius > 4) {
			radius = 4;
			printf("Radius too large, setting to 4.\n");
		}
		if(neighbourhood > 2) {
			radius = 2;
			printf("Neighbourhood too large, setting to 2.\n");
		}

		// create program
		if(NULL == ocl.ReadNLMProgram("NLMProgram.cl")) {
			printf("Cannot load program\n");
			return false;
		}
		cl_program program = clCreateProgramWithSource(ocl.context, 1, (const char**) &ocl.szNLMProgram, NULL, NULL);
		if(program == NULL) {
			printf("Cannot create program\n");
			return false;
		}

		// bulid program
		int errcode = clBuildProgram(program, 1, &(ocl.device), NULL, NULL, NULL);
		if (errcode != CL_SUCCESS) {
			size_t len;
			char buffer[10000];
			printf("Error: Failed to build program executable (%d)\n", errcode);            
			clGetProgramBuildInfo(program, ocl.device, CL_PROGRAM_BUILD_LOG,
											  sizeof(buffer), buffer, &len);
			printf("%s\n", buffer);
			return false;
		}

		// create kenerl
		cl_kernel kerNLM = clCreateKernel(program, "NLMeansOptV2", NULL);
		cl_kernel kerLocMeanVar = clCreateKernel(program, "LocMeanAndVar", NULL);
		cl_kernel kerPseudoRes = clCreateKernel(program, "PseudoRes", NULL);
//		cl_kernel kerMeanVar = clCreateKernel(program, "LocMeanAndVar", NULL);

		copySize(src);
		copyOrigPos(src);

		std::vector<int> vStartX, vStartY, vStartZ;
		std::vector<int> vWidthX, vWidthY, vWidthZ;
		
		int border = radius + neighbourhood;

		const int BLOCK_NLOPT=13;

		const int BLOCK_WEIGHTS=9;

		int xkern = BLOCK_NLOPT - 2*border; // size of "kernel", i.e. voxels that are actually computed in X direction
		int ykern = BLOCK_NLOPT - 2*border; // size of "kernel", i.e. voxels that are actually computed in Y direction

		SPoint3D<int> blockSize, blocks;
		blocks.init(std::min(100, (width + xkern - 1)/xkern), std::min(100, (height + ykern - 1)/ykern), depth);
		blockSize.init(blocks.x * xkern, blocks.y * ykern, std::min(100, depth));

		int offset;
		offset = 0;
		while(offset < src.width) {
			vStartX.push_back(offset);
			vWidthX.push_back(std::min(blockSize.x, src.width - offset));
			offset += blockSize.x;
		}	
		offset = 0;
		while(offset < src.height) {
			vStartY.push_back(offset);
			vWidthY.push_back(std::min(blockSize.y, src.height - offset));
			offset += blockSize.y;
		}	
		offset = 0;
		while(offset < src.depth) {
			vStartZ.push_back(offset);
			vWidthZ.push_back(std::min(blockSize.z, src.depth - offset));
			offset += blockSize.z;
		}	

		Timer timer;
		timer.restart();

		printf("Starting computation\n");
		printf("Preprocessing pseudoresiduals\n");

		int blocksTotal = (int)(vStartX.size() * vStartY.size() * vStartZ.size());
		int blocksDone = 0;
		double accum = 0;
		int accumVals = 0;
		for(int k = 0; k < (int)vStartZ.size(); k++) 
		for(int j = 0; j < (int)vStartY.size(); j++) 
		for(int i = 0; i < (int)vStartX.size(); i++) {
			printf("Computing Pseudoresiduals block %d/%d\n", ++blocksDone, blocksTotal);
			
			int sizeAsArray[4]; // size must be 4-component, because we use it as parameter for kernels
			SPoint3D<int> &size = *((SPoint3D<int>*)sizeAsArray);
			SPoint3D<int> orig;
			int pseudoresBorder = 1;
			orig.x = vStartX[i] - pseudoresBorder;
			orig.y = vStartY[j] - pseudoresBorder;
			orig.z = vStartZ[k] - pseudoresBorder;
			SPoint3D<int> gpuBlocks;
			gpuBlocks.init(vWidthX[i], vWidthY[j], vWidthZ[k]);
			size.x = gpuBlocks.x + 2*pseudoresBorder;
			size.y = gpuBlocks.y + 2*pseudoresBorder;
			size.z = gpuBlocks.z + 2*pseudoresBorder;

			int numEntries = size.x * size.y * size.z;
			int numPseudores = gpuBlocks.x * gpuBlocks.y * gpuBlocks.z;

			size_t iGlobSize = gpuBlocks.y * gpuBlocks.x; // one kernel for each line in X direction
			size_t iLocSize = gpuBlocks.x;

			float *pVolArray = new float[numEntries];
			float *pPseudoresArray = new float[numPseudores];

			memset(pPseudoresArray, 0, sizeof(float)*numPseudores);
			src.saveSubvolumeToArray(pVolArray, orig, size);

			// alloc memory
			cl_mem memSrc, memPseudores;
			memSrc = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*numEntries, pVolArray, NULL);
			memPseudores = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)*numPseudores, NULL, NULL);
			
			//set arguments
			clSetKernelArg(kerPseudoRes, 0, sizeof(cl_mem), (void*)&memSrc);
			clSetKernelArg(kerPseudoRes, 1, sizeof(cl_mem), (void*)&memPseudores);
			clSetKernelArg(kerPseudoRes, 2, sizeof(cl_int4), &size.x);
			
			// TODO: make pseudores faster
			errcode = clEnqueueNDRangeKernel(ocl.queue, kerPseudoRes, 1, NULL, &iGlobSize, NULL, 0, NULL, NULL);
			clEnqueueReadBuffer(ocl.queue, memPseudores, CL_TRUE, 0, sizeof(float) * numPseudores, pPseudoresArray, 0, NULL, NULL);
			clReleaseMemObject(memPseudores);
			clReleaseMemObject(memSrc);

			// compute variance from pseudoresidual
			int idx = 0;
			for(idx = 0; idx < numPseudores; idx++) {
				double val = (double) pPseudoresArray[idx];
				accum += val*val;
				accumVals++;
			}

			//loadSubvolumeFromArray(pPseudoresArray, SPoint3D<int>(vStartX[i],vStartY[j],vStartZ[k]), gpuBlocks, 0);
			delete[] pPseudoresArray;
			delete[] pVolArray;
		}

		float variance = (float)((double)accum / (double)(accumVals));
		int nbhsize = (neighbourhood*2)+1;
		nbhsize = nbhsize * nbhsize * nbhsize;
		float weightConst = (float) (2 * (float)beta * variance * (float)(nbhsize));

		blocksDone = 0;
		for(int k = 0; k < (int)vStartZ.size(); k++) 
		for(int j = 0; j < (int)vStartY.size(); j++) 
		for(int i = 0; i < (int)vStartX.size(); i++) {
			printf("Computing block %d/%d\n", ++blocksDone, blocksTotal);
			
			int sizeAsArray[4]; // size must be 4-component, because we use it as parameter for kernels
			SPoint3D<int> &size = *((SPoint3D<int>*)sizeAsArray);
			SPoint3D<int> orig;
			orig.x = vStartX[i] - border;
			orig.y = vStartY[j] - border;
			orig.z = vStartZ[k] - border;
			SPoint3D<int> gpuBlocks;
			gpuBlocks.init(vWidthX[i], vWidthY[j], 1);
			size.x = gpuBlocks.x + 2*border;
			size.y = gpuBlocks.y + 2*border;
			size.z = vWidthZ[k] + 2*border;

			int numEntries = size.x * size.y * size.z;

			float *pVolArray = new float[numEntries];

			memset(pVolArray, 0, sizeof(float)*numEntries);
			src.saveSubvolumeToArray(pVolArray, orig, size);

			cl_mem memSrc = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*numEntries, pVolArray, NULL);
			cl_mem memDest = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)*numEntries, NULL, NULL);
			
			cl_mem memMean = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)*numEntries, NULL, NULL);
			cl_mem memVar = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)*numEntries, NULL, NULL);

			printf("Computing local mean and variance...\n");
			clSetKernelArg(kerLocMeanVar, 0, sizeof(cl_mem), (void*)&memSrc);
			clSetKernelArg(kerLocMeanVar, 1, sizeof(cl_mem), (void*)&memMean);
			clSetKernelArg(kerLocMeanVar, 2, sizeof(cl_mem), (void*)&memVar);
			clSetKernelArg(kerLocMeanVar, 3, sizeof(cl_int), &radius);
			clSetKernelArg(kerLocMeanVar, 4, sizeof(cl_int4), &size.x);

			size_t globs[3], locs[3];
			locs[0] = size.y;
			globs[0] = size.z * locs[0];

			errcode = clEnqueueNDRangeKernel(ocl.queue, kerLocMeanVar, 1, NULL, globs, locs, 0, NULL, NULL);

			// HACK
			float fMeanAccept = 0.1f;
			float fSigma = 0.9999f;


			printf("Computing NM optimized filter...\n");
			//set arguments
			int idxParam = 0;
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_mem), (void*)&memSrc);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_mem), (void*)&memDest);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_mem), (void*)&memMean);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_mem), (void*)&memVar);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_int), &radius);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_int), &neighbourhood);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_float), &weightConst);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_float), &fMeanAccept);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_float), &variance);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_float), &fSigma);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_int4), &size.x);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_float) * BLOCK_NLOPT * BLOCK_NLOPT * BLOCK_NLOPT, NULL);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_float) * BLOCK_WEIGHTS * BLOCK_WEIGHTS * BLOCK_WEIGHTS, NULL);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_float) * BLOCK_WEIGHTS * BLOCK_WEIGHTS, NULL);
			clSetKernelArg(kerNLM, idxParam++, sizeof(cl_float) * BLOCK_WEIGHTS, NULL);
					
			locs[0] = BLOCK_NLOPT; 
			locs[1] = BLOCK_NLOPT;
			locs[2] = BLOCK_NLOPT;
			globs[0] = locs[0] * gpuBlocks.x;
			globs[1] = locs[1] * gpuBlocks.y;
			globs[2] = locs[2] * gpuBlocks.z;

			errcode = clEnqueueNDRangeKernel(ocl.queue, kerNLM, 2, NULL, globs, locs, 0, NULL, NULL);
			memset(pVolArray, 0, sizeof(float) * numEntries);
			clEnqueueReadBuffer(ocl.queue, memDest, CL_TRUE, 0, sizeof(float) * numEntries, pVolArray, 0, NULL, NULL);
			
			printf("Storing results...\n");
			loadSubvolumeFromArray(pVolArray, orig, size, border);

			delete[] pVolArray;

			clReleaseMemObject(memDest);
			clReleaseMemObject(memSrc);
			clReleaseMemObject(memMean);
			clReleaseMemObject(memVar);
		}
		
		timer.measure();
		int h,m,s,ms;
		timer.getTime(h, m, s, ms);
		printf("Time taken: %d:%d:%d.%3d\n", h,m,s,ms);
		return true;
	}

#endif

	template <class T2>
	friend bool volGetLocalMinima(const CVolumeSet<T2> &src, CVolumeSet<int> &markers);
	
	template <class T2>
	friend bool volWatershedBasic(CVolumeSet<T2> &dest, const CVolumeSet<T2> &src, const CVolumeSet<int> &markers);

	CVolumeSet(int sizeX, int sizeY, int sizeZ) {
		cutXZ = cutYZ = NULL;
		origPos.x = origPos.y = origPos.z = 0;
		realloc(sizeX, sizeY, sizeZ);
		//printf("tajp: %s\n", typeid(T).name());
	};

	~CVolumeSet() {
		reset();

	};
};


// helper functions:
// computes gauss lowpass filter in X direction (radius == 1 means no filtering, radius == 2 means filter [1,2,1], radius == 3 means [1,4,6,4,1], etc.)
void volGaussFilterX(CVolumeSet<float> &dest, const CVolumeSet<float> &src, float defaultValue, unsigned int radius);
// computes gauss lowpass filter in Y direction (radius == 1 means no filtering, radius == 2 means filter [1,2,1], radius == 3 means [1,4,6,4,1], etc.)
void volGaussFilterY(CVolumeSet<float> &dest, const CVolumeSet<float> &src, float defaultValue, unsigned int radius);
// computes gauss lowpass filter in Z direction (radius == 1 means no filtering, radius == 2 means filter [1,2,1], radius == 3 means [1,4,6,4,1], etc.)
void volGaussFilterZ(CVolumeSet<float> &dest, const CVolumeSet<float> &src, float defaultValue, unsigned int radius);
// computes gauss lowpass filter in all directions (radius == 1 means no filtering, radius == 2 means filter [1,2,1], radius == 3 means [1,4,6,4,1], etc.)
void volGaussFilter3D(CVolumeSet<float> &dest, const CVolumeSet<float> &src, float defaultValue, unsigned int radius);
// computes the gradient of the float volumeset
void volGradient(CVolumeSet<SPoint3D<float> > &dest, const CVolumeSet<float> &src);
// computes the size of gradient of the float volumeset
void volGradientSizeApprox(CVolumeSet<float> &dest, const CVolumeSet<float> &src);
// creates gauss filter of given radius (radius = 1 means [1], radius = 2 means [1,2,1], radius = 3 means [1,4,6,4,1])
std::vector<float> buildGaussFilter(unsigned int radius);
// outputs binary mask to disk (multiplied by given constant) as szFilename.txt and szFilename.raw
#ifndef ONLY_VIEWER
void dbgWriteMaskToDisk(const CVolumeSet<unsigned char> &mask, CImageSet &imgset, char *szFilename, unsigned char multiplier = 1);
#endif
// creates 12-bit test data of size 128x128z128 and a vector of border seeds
void createTestData(CVolumeSet<short> &result, std::list<SPoint3D<int> > &borderSeeds);

template<class T>
struct WatershedCell {
	T value;
	int idx;
	int x, y, z;
	WatershedCell(T value, int idx, int x, int y, int z) : value(value), idx(idx), x(x), y(y), z(z) {}
};

template<class T>
bool operator<(const WatershedCell<T> &c1, const WatershedCell<T> &c2) {
	return c1.value > c2.value; // inverse operator for giving smaller values first in the priority queue
}


template<class T>
void volGradientSizeApprox(CVolumeSet<T> &dest, const CVolumeSet<T> &src, const T &threshold) {
	dest.copySize(src);

	int width = src.getWidth();
	int height = src.getHeight();
	int depth = src.getDepth();

	int i, j, k;
	for(k = 0; k < depth; k++) {
		for(j = 0; j < height; j++) {
			for(i = 0; i < width; i++) {
				if((k == 0) || (k == depth - 1) || 
						(j == 0) || (j == height - 1) ||
						(i == 0) || (i == width - 1)) {
					dest.setValue(i,j,k, src.getValue(i,j,k));
				}else {
					T gradientX, gradientY, gradientZ;
					gradientX = (src.getValue(i + 1, j, k) - src.getValue(i-1, j, k)) / 2;
					gradientY = (src.getValue(i, j + 1, k) - src.getValue(i, j - 1, k)) / 2;
					gradientZ = (src.getValue(i, j, k + 1) - src.getValue(i, j, k - 1)) / 2;

					T gradSize = sqrt(gradientX * gradientX + gradientY * gradientY + gradientZ * gradientZ);
					if(gradSize <= threshold)
						gradSize = 0;
					dest.setValue(i,j,k, gradSize);
				}
			}
		}
	}
}

void getNeighbourIndices(std::vector<SPoint3D<int> > &destIndices) {
	destIndices.push_back(SPoint3D<int>(-1,-1,-1));
	destIndices.push_back(SPoint3D<int>( 0,-1,-1));
	destIndices.push_back(SPoint3D<int>( 1,-1,-1));

	destIndices.push_back(SPoint3D<int>(-1, 0,-1));
	destIndices.push_back(SPoint3D<int>( 0, 0,-1));
	destIndices.push_back(SPoint3D<int>( 1, 0,-1));

	destIndices.push_back(SPoint3D<int>(-1, 1,-1));
	destIndices.push_back(SPoint3D<int>( 0, 1,-1));
	destIndices.push_back(SPoint3D<int>( 1, 1,-1));

	destIndices.push_back(SPoint3D<int>(-1,-1, 0));
	destIndices.push_back(SPoint3D<int>( 0,-1, 0));
	destIndices.push_back(SPoint3D<int>( 1,-1, 0));

	destIndices.push_back(SPoint3D<int>(-1, 0, 0));
	//destIndices.push_back(SPoint3D<int>( 0, 0, 0));
	destIndices.push_back(SPoint3D<int>( 1, 0, 0));

	destIndices.push_back(SPoint3D<int>(-1, 1, 0));
	destIndices.push_back(SPoint3D<int>( 0, 1, 0));
	destIndices.push_back(SPoint3D<int>( 1, 1, 0));

	destIndices.push_back(SPoint3D<int>(-1,-1, 1));
	destIndices.push_back(SPoint3D<int>( 0,-1, 1));
	destIndices.push_back(SPoint3D<int>( 1,-1, 1));

	destIndices.push_back(SPoint3D<int>(-1, 0, 1));
	destIndices.push_back(SPoint3D<int>( 0, 0, 1));
	destIndices.push_back(SPoint3D<int>( 1, 0, 1));

	destIndices.push_back(SPoint3D<int>(-1, 1, 1));
	destIndices.push_back(SPoint3D<int>( 0, 1, 1));
	destIndices.push_back(SPoint3D<int>( 1, 1, 1));
}


#define MARKER_UNDEF -1
#define MARKER_VISITED -2
#define MARKER_BORDER -3
#define MARKER_VALID 0

template<class T>
bool volGetLocalMinima(/*const*/ CVolumeSet<T> &src, CVolumeSet<int> &markers) {
	markers.copySize(src);
	markers.setValue(MARKER_UNDEF);

	int width, depth, height;
	src.getSize(width, height, depth);

	int currentIdx = 0;
	int totalFilled = 0;

	std::vector<SPoint3D<int> > vNeighbours;
	getNeighbourIndices(vNeighbours);

	for(int z = 0; z < depth; z++) {
	for(int y = 0; y < height; y++) {
		int offset = width * y;

		std::list<WatershedCell<T> > cellQueue;
		std::list<WatershedCell<T> > fillQueue;
		for(int x = 0; x < width; x++) {
			T value = src.planes[z][offset+x];

			if(markers.planes[z][offset+x] == MARKER_UNDEF) {
				// found a voxel, that has not been processed yet
				cellQueue.push_back(WatershedCell<T>(value, currentIdx, x, y, z));
				markers.planes[z][offset+x] = MARKER_VISITED;

				bool bMinimum = true;

				// floodfill
				while(!cellQueue.empty()) {
					WatershedCell<T> cell = *cellQueue.begin();
					cellQueue.pop_front();
					fillQueue.push_back(cell);

					int cellOffset = width * cell.y + cell.x;
					
					int tx, ty, tz;
					// visit all neighbours
					for(int n = 0; n < (int)vNeighbours.size(); n++) {
						tx = cell.x + vNeighbours[n].x;
						ty = cell.y + vNeighbours[n].y;
						tz = cell.z + vNeighbours[n].z;
						if(tx < 0 || ty < 0 || tz < 0 || tx >= width || ty >= height || tz >= depth)
							continue;

						T destValue = src.getValue(tx, ty, tz);
						if(destValue < value) {
							// a lower value has been found on the border -> this is not a minimum (but continue to fill, to prevent false minima)
							bMinimum = false;
						} else if(destValue == value && (markers.getValue(tx, ty, tz) == MARKER_UNDEF)) {
							// another possible member of a minimum, add to queue
							cellQueue.push_back(WatershedCell<T>(value, currentIdx, tx, ty, tz));
							markers.setValue(tx, ty, tz, MARKER_VISITED);
						} 
					}
				}

				// found true minimum, paint found voxels with marker id
				if(bMinimum) { 
					totalFilled += (int)fillQueue.size();
					typename std::list<WatershedCell<T> >::iterator itr;
					for(itr = fillQueue.begin(); itr != fillQueue.end(); itr++) {
						WatershedCell<T> cell = *itr;

						ASSERT(markers.planes[cell.z][cell.y * width + cell.x] >= 0);
							//printf("Voxel already marked! (%d, %d, %d) id=%d newid=%d\n", cell.x, cell.y, cell.z, markers.planes[cell.z][cell.y * width + cell.x], currentIdx);
						markers.planes[cell.z][cell.y * width + cell.x] = currentIdx;
					}
					currentIdx++;
				}

				cellQueue.clear();
				fillQueue.clear();
			}
		}
	}
	}

	printf("Total number of markers: %d\n", currentIdx);
	printf("Total number of marked voxels: %d\n", totalFilled);

	int cleanedVoxels = 0;

	// clear temporary VISITED flag
	for(int k = 0; k < depth; k++) {
	for(int j = 0; j < height; j++) {
		int offset = width * j;

		for(int i = 0; i < width; i++) {
			if(markers.planes[k][offset + i] == MARKER_VISITED)
				markers.planes[k][offset + i] = MARKER_UNDEF;

			// set all voxels on borders to undef marker, otherwise it is connecting our segmented watersheds
			if(i == 0 || i == width-1 || j == 0 || j == height-1 || k == 0 || k == depth-1) {
				markers.planes[k][offset+i] = MARKER_UNDEF;
				cleanedVoxels++;
			}

//			if(markers.getValue(i,j,k) != MARKER_UNDEF)
//				src.setValue(i,j,k, 3000);

			
		}
	}
	}

	printf("Cleaned voxels on border = %d\n", cleanedVoxels);

	return true;
}

template<class T>
bool volWatershedBasic(CVolumeSet<T> &dest, const CVolumeSet<T> &src, const CVolumeSet<T> &orig, CVolumeSet<int> &markers) {
	if(!(src.getSize() == markers.getSize() && (src.getSize() == orig.getSize() ))) {
		printf("Markers not the same size as source volume.");
		return false;
	}
	dest.copyVolume(orig);
	// construct queue
	std::priority_queue<WatershedCell<T> > processQueue;

	int width, depth, height;
	src.getSize(width, height, depth);

	// construct a field of +-1 indices for indexing neighbours
	std::vector<SPoint3D<int> > vNeighbours;
	getNeighbourIndices(vNeighbours);

	int markerBorders = 0;
	int markerVoxels = 0;
	for(int k = 0; k < depth; k++) {
	for(int j = 0; j < height; j++) {
		int offset = width * j;

		for(int i = 0; i < width; i++) {
			int markerIdx = markers.getValue(i,j,k);
			if(markerIdx >= MARKER_VALID) {
				markerVoxels++;
				bool bBorder = false;
				// find border voxels on defined markers, use them as seeds to priority queue
				for(int n = 0; n < (int)vNeighbours.size(); n++) {
					int tx = i + vNeighbours[n].x;
					int ty = j + vNeighbours[n].y;
					int tz = k + vNeighbours[n].z;
					if(tx < 0 || ty < 0 || tz < 0 || tx >= width || ty >= height || tz >= depth)
						continue;

					if(markers.getValue(tx, ty, tz) == MARKER_UNDEF) 
						bBorder = true;
				}

				if(bBorder) {
					markerBorders ++;
					processQueue.push(WatershedCell<T>(src.getValue(i,j,k), markerIdx, i, j, k));
				}

				// testing - each two regions of local minima should not be neighbours
				bool bNeighbours = false;
				for(int n = 0; n < (int)vNeighbours.size(); n++) {
					int tx = i + vNeighbours[n].x;
					int ty = j + vNeighbours[n].y;
					int tz = k + vNeighbours[n].z;
					if(tx < 0 || ty < 0 || tz < 0 || tx >= width || ty >= height || tz >= depth)
						continue;

					T value = markers.getValue(tx, ty, tz);
					if(value >= MARKER_VALID && value != markerIdx) 
						bNeighbours = true;
				}

				if (bNeighbours) {
					printf("Neighbouring markers! at (%d, %d, %d)", i, j, k);
				}
				
			}
		}
	}
	}

	printf("Marker Voxels = %d\nMarker Borders = %d", markerVoxels, markerBorders);

	int totalProcessed = 0;
	int mark = 1000;
	while(!processQueue.empty()) {
		// get smallest value
		const WatershedCell<T> cell = processQueue.top();
		processQueue.pop();
		totalProcessed++;
		if(totalProcessed > mark) {
			printf(".");
			mark += 1000;
		}

		int i,j,k;
		i = cell.x;
		j = cell.y;
		k = cell.z;
		int offset = width * j;
		int markerIdx = cell.idx;

		markers.setValue(i,j,k, cell.idx);

		bool bBorder = false;

		for(int n = 0; n < (int) vNeighbours.size(); n++) {
			int tx = i + vNeighbours[n].x;
			int ty = j + vNeighbours[n].y;
			int tz = k + vNeighbours[n].z;
			if(tx < 0 || ty < 0 || tz < 0 || tx >= width || ty >= height || tz >= depth)
				continue;

			int destMarker = markers.getValue(tx, ty, tz);
			if(destMarker == MARKER_UNDEF) {
				processQueue.push(WatershedCell<T>(src.getValue(tx, ty, tz), markerIdx, tx, ty, tz));
				//markers.setValue(tx, ty, tz, markerIdx);
				markers.setValue(tx,ty,tz, MARKER_VISITED);
			} else if(destMarker >= MARKER_VALID && destMarker != MARKER_BORDER && destMarker != markerIdx) {
				bBorder = true;
			}
		}
		if(bBorder)
			markers.setValue(i,j,k, MARKER_BORDER);
	}


	for(int z = 0; z < depth; z++) {
	for(int y = 0; y < height; y++) {
	for(int x = 0; x < width; x++) {
		if(markers.getValue(x,y,z) == MARKER_BORDER)
			dest.setValue(x,y,z, 3000);
		/*else 
			dest.setValue(x,y,z, -1000);*/
	}
	}
	}

	printf("number of voxels: %d\n", totalProcessed);

	return true;
}

template<class T>
void volCreateTestData(CVolumeSet<T> &dest) {
	int width, height, depth;
	dest.getSize(width, height, depth);
	
	srand(time(NULL));

	std::vector<SPoint3D<int> > centers;
//	SPoint3D<int> center(width/2, height/2, depth/2);
//	SPoint3D<int> center2(width/2 + width/4, height/2, depth/2);

	for(int v = 0; v < 10; v++) {
		centers.push_back(SPoint3D<int>(rand()%width, rand()%height, rand()%depth));
	}
	
	for(int k = 0; k < depth; k++) {
		for(int j = 0; j < height; j++) {
			for(int i = 0; i < width; i++) {
				T value;
				for(int v = 0; v < centers.size(); v++) {
					T newValue = (centers[v].x-i)*(centers[v].x-i) + (centers[v].y-j)*(centers[v].y-j) + (centers[v].z-k)*(centers[v].z-k);
					if(v == 0 || newValue < value)
						value = newValue;
				}
				dest.setValue(i, j, k, value);
			}
		}
	}
		
}

}

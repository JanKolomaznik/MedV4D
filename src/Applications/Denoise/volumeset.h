/***********************************
	CVOLUMESET
************************************/

#include <omp.h>

namespace viewer {

class CInfoDialog;
class CProgress;

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
	imax = (int)min(vec1.size(), vec2.size());
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
	imax = (int)min(vec1.size(), vec2.size());
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
	bool saveToDisk(wchar_t *wsFilename, CInfoDialog *info) {
		FILE *fw;
		if(NULL == (fw = _wfopen(wsFilename, L"wb"))) {
			std::wstring errstr;
			errstr = std::wstring(L"Could not create file: ") + std::wstring(wsFilename);
			info->setMessage(errstr);
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
		strncpy(buff, name,31);
		fwrite(name, sizeof(char), 32, fw);
		
		for(int k = 0; k < depth; k++)
		{
			fwrite(planes[k], sizeof(T), width*height, fw);
		}

		fclose(fw);
		return true;
	}

	// loads volumeset from disk
	void loadFromDisk(wchar_t *wsFilename, CInfoDialog *info) {
		FILE *fr;
		if(NULL == (fr = _wfopen(wsFilename, L"rb"))) {
			std::wstring errstr;
			errstr = std::wstring(L"Could not open file: ") + std::wstring(wsFilename);
			if(info)
				info->setMessage(errstr);
			return;
		}

		unsigned char header[3];
		fread(header, sizeof(unsigned char), 3, fr);
		if(header[0] != 'V' || header[1] != 'o' || header[2] != 'l')
		{
			std::wstring errstr;
			errstr = std::wstring(L"Wrong file type: ") + std::wstring(wsFilename);
			if(info)
				info->setMessage(errstr);
			fclose(fr);
			return;
		}

		unsigned int size[3];
		fread(size, sizeof(unsigned int), 3, fr);
		width = size[0];
		height = size[1];
		depth = size[2];

		char name[32];
		fread(name, sizeof(char), 32, fr);
		if(0 != strcmp(name, typeid(T).name()))
		{
			std::wstring errstr;
			errstr = std::wstring(L"Wrong file type (") + str2wcs(typeid(T).name()) + L"): " + wsFilename + L"(" + str2wcs(name) + L")";
			if(info)
				info->setMessage(errstr);
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

		std::list<SPoint2D<unsigned int>> queue;
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

		std::list<SPoint2D<unsigned int>> queue;
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

		std::list<SPoint3D<unsigned int>> queue;
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

		radius = max(0,radius);

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
				__int64 voxels = 0;
				for(sk = max(k - radius, 0); sk <= min(k + radius, depth-1); sk++)
				for(sj = max(j - radius, 0); sj <= min(j + radius, height-1); sj++)
				for(si = max(i - radius, 0); si <= min(i + radius, width-1); si++) {
					accum += src.getValue(si,sj,sk);
					voxels++;
				}
				double mean = accum / (double)voxels;

				accum = 0;
				// compute sqrt(var)
				for(sk = max(k - radius, 0); sk <= min(k + radius, depth-1); sk++)
				for(sj = max(j - radius, 0); sj <= min(j + radius, height-1); sj++)
				for(si = max(i - radius, 0); si <= min(i + radius, width-1); si++) {
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

		radius = max(0,radius);

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
				__int64 voxels = 0;
				for(sk = max(k - radius, 0); sk <= min(k + radius, depth-1); sk++)
				for(sj = max(j - radius, 0); sj <= min(j + radius, height-1); sj++)
				for(si = max(i - radius, 0); si <= min(i + radius, width-1); si++) {
					accum += src.getValue(si,sj,sk);
					voxels++;
				}
				double mean = accum / (double)voxels;

				accum = 0;
				// compute sqrt(var)
				for(sk = max(k - radius, 0); sk <= min(k + radius, depth-1); sk++)
				for(sj = max(j - radius, 0); sj <= min(j + radius, height-1); sj++)
				for(si = max(i - radius, 0); si <= min(i + radius, width-1); si++) {
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

		radius = max(0,radius);

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
				__int64 voxels = 0;
				for(sk = max(k - radius, 0); sk <= min(k + radius, depth-1); sk++)
				for(sj = max(j - radius, 0); sj <= min(j + radius, height-1); sj++)
				for(si = max(i - radius, 0); si <= min(i + radius, width-1); si++) {
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
				histo[max(0, min(iVals - 1, convVal))]++;
			}
		}

		minValue = iMin;
		maxValue = iMax;
	}

	// performs a NL means filter on src, computed on +-radius and local neighborhoods are size +-neighborhood
	void volBasicNLMeans(const CVolumeSet<T> &src, T h, int radius, int neighborhood, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		radius = max(0,radius);

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
					int skmin = max(0,k - radius), skmax = min(depth-1,k+radius);
					int sjmin = max(0,j - radius), sjmax = min(height-1,j+radius);
					int simin = max(0,i - radius), simax = min(width-1,i+radius);

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

		radius = max(0,radius);

		unsigned int finished = 0;
		unsigned int last_update = 0; 

		progress->UpdateProgress(0);

		#pragma omp parallel for //num_threads(1)
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
					int skmin = max(0,k - radius), skmax = min(depth-1,k+radius);
					int sjmin = max(0,j - radius), sjmax = min(height-1,j+radius);
					int simin = max(0,i - radius), simax = min(width-1,i+radius);

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

		radius = max(0,radius);

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
					int skmin = max(0,k - radius), skmax = min(depth-1,k+radius);
					int sjmin = max(0,j - radius), sjmax = min(height-1,j+radius);
					int simin = max(0,i - radius), simax = min(width-1,i+radius);

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

	// performs a blockwise NL means filter on src
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

		radius = max(0,radius);

		unsigned int finished = 0;
		unsigned int last_update = 0; 
		//exp_fast_prepare();

		progress->UpdateProgress(0);

		#pragma omp parallel for 
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
					int skmin = max(0,k - radius), skmax = min(depth-1,k+radius);
					int sjmin = max(0,j - radius), sjmax = min(height-1,j+radius);
					int simin = max(0,i - radius), simax = min(width-1,i+radius);

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

					skmin = max(0, k - blockNbh); skmax = min(depth-1, k + blockNbh);
					sjmin = max(0, j - blockNbh); sjmax = min(height-1, j + blockNbh);
					simin = max(0, i - blockNbh); simax = min(width-1, i + blockNbh);
					
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

	void volMedianFilter(const CVolumeSet<T> &src, int radius, CProgress *progress) {
		copySize(src);
		copyOrigPos(src);

		// neighborhood size
		int nbhsize = (2*radius + 1);
		nbhsize = nbhsize * nbhsize * nbhsize;

		unsigned int finished = 0;
		unsigned int last_update = 0; 

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

	CVolumeSet(int sizeX, int sizeY, int sizeZ) {
		cutXZ = cutYZ = NULL;
		origPos.x = origPos.y = origPos.z = 0;
		realloc(sizeX, sizeY, sizeZ);
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
void volGradient(CVolumeSet<SPoint3D<float>> &dest, const CVolumeSet<float> &src);
// computes the size of gradient of the float volumeset
void volGradientSizeApprox(CVolumeSet<float> &dest, const CVolumeSet<float> &src);
// creates gauss filter of given radius (radius = 1 means [1], radius = 2 means [1,2,1], radius = 3 means [1,4,6,4,1])
std::vector<float> buildGaussFilter(unsigned int radius);
// outputs binary mask to disk (multiplied by given constant) as szFilename.txt and szFilename.raw
#ifndef ONLY_VIEWER
void dbgWriteMaskToDisk(const CVolumeSet<unsigned char> &mask, CImageSet &imgset, char *szFilename, unsigned char multiplier = 1);
#endif
// creates 12-bit test data of size 128x128z128 and a vector of border seeds
void createTestData(CVolumeSet<short> &result, std::list<SPoint3D<int>> &borderSeeds);




}
#include "hapticCursor.h"

void M4D::Viewer::hapticCursor::startHaptics()
{
	numHapticDevices = handler->getNumDevices();
	LOG("Chai3d: getNumDevices = ", numHapticDevices);

	if (numHapticDevices > 0)
	{
		LOG("Chai3d: using hatpic device #0");
		handler->getDevice(hapticDevice, 0);

		// open connection to haptic device
		LOG("Chai3d: open device");
		hapticDevice->open();

		// initialize haptic device
		LOG("Chai3d: init device");
		hapticDevice->initialize();

		// retrieve information about the current haptic device
		info = hapticDevice->getSpecifications();
		LOG("Chai3d: ", info.m_manufacturerName, " ", info.m_modelName);

		if (_inPort->GetDatasetTyped().GetDimension() == 3)
		{
			_minX = _inPort->GetDatasetTyped().GetDimensionExtents(0).minimum;
			_minY = _inPort->GetDatasetTyped().GetDimensionExtents(1).minimum;
			_minZ = _inPort->GetDatasetTyped().GetDimensionExtents(2).minimum;

			_sizeX = _inPort->GetDatasetTyped().GetDimensionExtents(0).maximum - _minX;
			_sizeY = _inPort->GetDatasetTyped().GetDimensionExtents(1).maximum - _minY;
			_sizeZ = _inPort->GetDatasetTyped().GetDimensionExtents(2).maximum - _minZ;

			maxSize = _sizeX > _sizeY ? _sizeX : _sizeY;
			maxSize = maxSize > _sizeZ ? maxSize : _sizeZ;
		}

		position.zero();

		_imageID = _inPort->GetDatasetTyped().GetElementTypeID();
		uint64 min = MAX_INT64;
		uint64 max = 0;
		uint64 result = 0;
		for (int i = _minX; i < _minX + _sizeX; i++)
		{
			for (int j = _minY; j < _minY + _sizeY; j++)
			{
				for (int k = _minZ; k < _minZ + _sizeZ; k++)
				{
					NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
						_imageID, result = Imaging::Image< TTYPE, 3 >::CastAImage(_inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >(i, j, k) ) );
					if (result > max)
					{
						max = result;
					}
					if (result < min)
					{
						min = result;
					}
				}
			}
		}
		_minValue = min;
		_maxValue = max;

		hapticsThread = new cThread();
		hapticsThread->set(&M4D::Viewer::hapticCursor::updateHaptics, CHAI_THREAD_PRIORITY_HAPTICS);

		LOG("Haptics thread started.");
	}
}

void M4D::Viewer::hapticCursor::updateHaptics()
{
	cVector3d newPosition;
	cVector3d track;
	cVector3d force;
	float fforce;
	uint64 result = 0;

	while (runHpatics)
	{
		bool outx, outy, outz = false;
		hapticDevice->getPosition(newPosition);
		track = newPosition - position;
		newPosition /= info.m_workspaceRadius;
		position = newPosition;
		newPosition *= maxSize;
		track.normalize();
		force = track;
		force.negate();

		if ((newPosition.x > (_sizeX / 2.0)) || (newPosition.x < _minX))
		{
			outx = true;
		}
		if ((newPosition.y > (_sizeY / 2.0)) || (newPosition.y < _minY))
		{
			outy = true;
		}
		if ((newPosition.z > (_sizeZ / 2.0)) || (newPosition.z < _minZ))
		{
			outz = true;
		}

		if (outx || outy || outz)
		{
			hapticDevice->setForce(force * info.m_maxForce);
		}
		else
		{
			NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
				_imageID, result = Imaging::Image< TTYPE, 3 >::CastAImage(_inPort->GetDatasetTyped()).GetElement( CreateVector< int32 >((int)newPosition.x, (int)newPosition.y, (int)newPosition.z) ) );
			fforce = result / _maxValue;
			hapticDevice->setForce(force * fforce * info.m_maxForce);
		}
		x = position.x;
		y = position.y;
		z = position.z;
	}
}
#pragma once


#include "MedV4D/Imaging/AImage.h"
#include <boost/filesystem.hpp>
#include <prognot/prognot.hpp>

M4D::Imaging::AImage::Ptr
loadItkImage(const boost::filesystem::path &aFile, prognot::ProgressNotifier aProgressNotifier);


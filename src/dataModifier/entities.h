/**
 *  @file entities.h
 *  @brief Definition supporting structures
 *  @author Vaclav Klecanda
 */

#ifndef ENTITIES_H
#define ENTITIES_H

/**
 *  @addtogroup datamodifier Data modifier( support utility)
 *  @{
 */

namespace M4D 
{
namespace DataModifier 
{

/// Structure for patient.
struct PatientInfo
{
  std::string patName;
  std::string patSex;
  std::string born;
};

/// Study informations.
struct StudyInfo
{
  std::string date;
};

}
}

/** @} */

#endif

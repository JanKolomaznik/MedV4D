#ifndef ENTITIES_H
#define ENTITIES_H

/**
 * Structures for patient & Study informations.
 */
namespace M4D {
namespace DataModifier {

struct PatientInfo
{
  std::string patName;
  std::string patSex;
  std::string born;
};

struct StudyInfo
{
  std::string date;
};

}
}

#endif

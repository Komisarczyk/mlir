
#ifndef _DETECTION_H_
#define _DETECTION_H_
#include "builders.h"
#include "matchers.h"
#include "scop.h"
isl::schedule_node rebuild(isl::schedule_node node,
                           const builders::ScheduleNodeBuilder &replacement);
isl::schedule_node
replaceOnce(isl::schedule_node node,
            const matchers::ScheduleNodeMatcher &pattern,
            const builders::ScheduleNodeBuilder &replacement);
isl::schedule_node
replaceDFSPreorderOnce(isl::schedule_node node,
                       const matchers::ScheduleNodeMatcher &pattern,
                       const builders::ScheduleNodeBuilder &replacement);

isl::schedule runDetection(pet::Scop &scop);

#endif
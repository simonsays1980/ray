// Copyright 2017 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

#include "ray/common/id.h"
#include "ray/common/ray_object.h"
#include "ray/common/task/task_spec.h"
#include "ray/raylet_client/raylet_client.h"
#include "ray/util/util.h"
#include "src/ray/protobuf/common.pb.h"

namespace ray {
namespace core {

using WorkerType = rpc::WorkerType;

// Return a string representation of the worker type.
std::string WorkerTypeString(WorkerType type);

// Return a string representation of the language.
std::string LanguageString(Language language);

// Return a string representation of the named actor to cache, in format of
// `namespace-[job_id-]actor_name`
std::string GenerateCachedActorName(const std::string &ns, const std::string &actor_name);

void SerializeReturnObject(const ObjectID &object_id,
                           const std::shared_ptr<RayObject> &return_object,
                           rpc::ReturnObject *return_object_proto);

/// Information about a remote function.
class RayFunction {
 public:
  RayFunction() {}
  RayFunction(Language language, const FunctionDescriptor &function_descriptor)
      : language_(language), function_descriptor_(function_descriptor) {}

  Language GetLanguage() const { return language_; }

  const FunctionDescriptor &GetFunctionDescriptor() const { return function_descriptor_; }

 private:
  Language language_;
  FunctionDescriptor function_descriptor_;
};

/// Options for all tasks (actor and non-actor) except for actor creation.
struct TaskOptions {
  TaskOptions() {}
  TaskOptions(std::string name,
              int num_returns,
              std::unordered_map<std::string, double> &resources,
              const std::string &concurrency_group_name = "",
              int64_t generator_backpressure_num_objects = -1,
              const std::string &serialized_runtime_env_info = "{}",
              bool enable_task_events = kDefaultTaskEventEnabled,
              const std::unordered_map<std::string, std::string> &labels = {})
      : name(name),
        num_returns(num_returns),
        resources(resources),
        concurrency_group_name(concurrency_group_name),
        serialized_runtime_env_info(serialized_runtime_env_info),
        generator_backpressure_num_objects(generator_backpressure_num_objects),
        enable_task_events(enable_task_events),
        labels(labels) {}

  /// The name of this task.
  std::string name;
  /// Number of returns of this task.
  int num_returns = 1;
  /// Resources required by this task.
  std::unordered_map<std::string, double> resources;
  /// The name of the concurrency group in which this task will be executed.
  std::string concurrency_group_name;
  /// Runtime Env Info used by this task. It includes Runtime Env and some
  /// fields which not contained in Runtime Env, such as eager_install.
  /// Propagated to child actors and tasks.
  std::string serialized_runtime_env_info;
  /// Only applicable when streaming generator is used.
  /// -1 means either streaming generator is not used or
  /// it is used but the feature is disabled.
  int64_t generator_backpressure_num_objects;
  /// True if task events (worker::TaskEvent) from this task should be reported, default
  /// to true.
  bool enable_task_events = kDefaultTaskEventEnabled;
  std::unordered_map<std::string, std::string> labels;
};

/// Options for actor creation tasks.
struct ActorCreationOptions {
  ActorCreationOptions() {}
  ActorCreationOptions(int64_t max_restarts,
                       int64_t max_task_retries,
                       int max_concurrency,
                       const std::unordered_map<std::string, double> &resources,
                       const std::unordered_map<std::string, double> &placement_resources,
                       const std::vector<std::string> &dynamic_worker_options,
                       std::optional<bool> is_detached,
                       std::string &name,
                       std::string &ray_namespace,
                       bool is_asyncio,
                       const rpc::SchedulingStrategy &scheduling_strategy,
                       const std::string &serialized_runtime_env_info = "{}",
                       const std::vector<ConcurrencyGroup> &concurrency_groups = {},
                       bool execute_out_of_order = false,
                       int32_t max_pending_calls = -1,
                       bool enable_task_events = kDefaultTaskEventEnabled,
                       const std::unordered_map<std::string, std::string> &labels = {})
      : max_restarts(max_restarts),
        max_task_retries(max_task_retries),
        max_concurrency(max_concurrency),
        resources(resources),
        placement_resources(placement_resources.empty() ? resources
                                                        : placement_resources),
        dynamic_worker_options(dynamic_worker_options),
        is_detached(std::move(is_detached)),
        name(name),
        ray_namespace(ray_namespace),
        is_asyncio(is_asyncio),
        serialized_runtime_env_info(serialized_runtime_env_info),
        concurrency_groups(concurrency_groups.begin(), concurrency_groups.end()),
        execute_out_of_order(execute_out_of_order),
        max_pending_calls(max_pending_calls),
        scheduling_strategy(scheduling_strategy),
        enable_task_events(enable_task_events),
        labels(labels) {
    // Check that resources is a subset of placement resources.
    for (auto &resource : resources) {
      auto it = this->placement_resources.find(resource.first);
      RAY_CHECK(it != this->placement_resources.end());
      RAY_CHECK_GE(it->second, resource.second);
    }
  };

  /// Maximum number of times that the actor should be restarted if it dies
  /// unexpectedly. A value of -1 indicates infinite restarts. If it's 0, the
  /// actor won't be restarted.
  const int64_t max_restarts = 0;
  /// Maximum number of times that individual tasks can be retried at the
  /// actor, if the actor dies unexpectedly. If -1, then the task may be
  /// retried infinitely many times.
  const int64_t max_task_retries = 0;
  /// The max number of concurrent tasks to run on this direct call actor.
  const int max_concurrency = 1;
  /// Resources required by the whole lifetime of this actor.
  const std::unordered_map<std::string, double> resources;
  /// Resources required to place this actor.
  const std::unordered_map<std::string, double> placement_resources;
  /// The dynamic options used in the worker command when starting a worker process for
  /// an actor creation task.
  const std::vector<std::string> dynamic_worker_options;
  /// Whether to keep the actor persistent after driver exit. If true, this will set
  /// the worker to not be destroyed after the driver shutdown.
  std::optional<bool> is_detached;
  /// The name to give this detached actor that can be used to get a handle to it from
  /// other drivers. This must be globally unique across the cluster.
  /// This should set if and only if is_detached is true.
  const std::string name;
  /// The namespace to give this detached actor so that the actor is only visible
  /// with the namespace.
  /// This should set if and only if is_detached is true.
  const std::string ray_namespace;
  /// Whether to use async mode of direct actor call.
  const bool is_asyncio = false;
  /// Runtime Env Info used by this task. It includes Runtime Env and some
  /// fields which not contained in Runtime Env, such as eager_install.
  /// Propagated to child actors and tasks.
  std::string serialized_runtime_env_info;
  /// The actor concurrency groups to indicate how this actor perform its
  /// methods concurrently.
  const std::vector<ConcurrencyGroup> concurrency_groups;
  /// Wether the actor execute tasks out of order.
  const bool execute_out_of_order = false;
  /// The maxmium actor call pending count.
  const int max_pending_calls = -1;
  // The strategy about how to schedule this actor.
  rpc::SchedulingStrategy scheduling_strategy;
  /// True if task events (worker::TaskEvent) from this creation task should be reported
  /// default to true.
  const bool enable_task_events = kDefaultTaskEventEnabled;
  const std::unordered_map<std::string, std::string> labels;
};

using PlacementStrategy = rpc::PlacementStrategy;

struct PlacementGroupCreationOptions {
  PlacementGroupCreationOptions(
      std::string name,
      PlacementStrategy strategy,
      std::vector<std::unordered_map<std::string, double>> bundles,
      bool is_detached,
      double max_cpu_fraction_per_node,
      NodeID soft_target_node_id = NodeID::Nil())
      : name(std::move(name)),
        strategy(strategy),
        bundles(std::move(bundles)),
        is_detached(is_detached),
        max_cpu_fraction_per_node(max_cpu_fraction_per_node),
        soft_target_node_id(soft_target_node_id) {
    RAY_CHECK(soft_target_node_id.IsNil() || strategy == PlacementStrategy::STRICT_PACK)
        << "soft_target_node_id only works with STRICT_PACK now";
  }

  /// The name of the placement group.
  const std::string name;
  /// The strategy to place the bundle in Placement Group.
  const PlacementStrategy strategy = rpc::PACK;
  /// The resource bundles in this placement group.
  const std::vector<std::unordered_map<std::string, double>> bundles;
  /// Whether to keep the placement group persistent after its creator dead.
  const bool is_detached = false;
  /// The maximum fraction of CPU cores this placement group can take up on each node.
  const double max_cpu_fraction_per_node;
  /// ID of the target node where bundles should be placed
  /// iff the target node has enough available resources and alive.
  /// Otherwise, the bundles can be placed elsewhere.
  /// Nil means there is no target node.
  /// This only applies to STRICT_PACK pg.
  const NodeID soft_target_node_id;
};

class ObjectLocation {
 public:
  ObjectLocation(NodeID primary_node_id,
                 int64_t object_size,
                 std::vector<NodeID> node_ids,
                 bool is_spilled,
                 std::string spilled_url,
                 NodeID spilled_node_id,
                 bool did_spill)
      : primary_node_id_(primary_node_id),
        object_size_(object_size),
        node_ids_(std::move(node_ids)),
        is_spilled_(is_spilled),
        spilled_url_(std::move(spilled_url)),
        spilled_node_id_(spilled_node_id),
        did_spill_(did_spill) {}

  const NodeID &GetPrimaryNodeID() const { return primary_node_id_; }

  const int64_t GetObjectSize() const { return object_size_; }

  const std::vector<NodeID> &GetNodeIDs() const { return node_ids_; }

  bool IsSpilled() const { return is_spilled_; }

  const std::string &GetSpilledURL() const { return spilled_url_; }

  const NodeID &GetSpilledNodeID() const { return spilled_node_id_; }

  const bool GetDidSpill() const { return did_spill_; }

 private:
  /// The ID of the node has the primary copy of the object.
  /// Nil if the object is pending resolution.
  const NodeID primary_node_id_;
  /// The size of the object in bytes. -1 if unknown.
  const int64_t object_size_;
  /// The IDs of the nodes that this object appeared on or was evicted by.
  const std::vector<NodeID> node_ids_;
  /// Whether this object has been spilled.
  const bool is_spilled_;
  /// If spilled, the URL of this object's spill location.
  const std::string spilled_url_;
  /// If spilled, the ID of the node that spilled the object. Nil if the object was
  /// spilled to distributed external storage.
  const NodeID spilled_node_id_;
  /// Whether or not this object was spilled.
  const bool did_spill_;
};

}  // namespace core
}  // namespace ray

namespace std {
template <>
struct hash<ray::rpc::LineageReconstructionTask> {
  size_t operator()(const ray::rpc::LineageReconstructionTask &task) const {
    size_t hash = std::hash<std::string>()(task.name());
    hash ^= std::hash<ray::rpc::TaskStatus>()(task.status());
    for (const auto &label : task.labels()) {
      hash ^= std::hash<std::string>()(label.first);
      hash ^= std::hash<std::string>()(label.second);
    }
    return hash;
  }
};
}  // namespace std

namespace ray {
namespace rpc {
inline bool operator==(const ray::rpc::LineageReconstructionTask &lhs,
                       const ray::rpc::LineageReconstructionTask &rhs) {
  return google::protobuf::util::MessageDifferencer::Equivalent(lhs, rhs);
}
}  // namespace rpc
}  // namespace ray

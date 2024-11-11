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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "ray/common/runtime_env_manager.h"
#include "ray/gcs/gcs_server/gcs_function_manager.h"
#include "ray/gcs/gcs_server/gcs_init_data.h"
#include "ray/gcs/gcs_server/gcs_table_storage.h"
#include "ray/gcs/pubsub/gcs_pub_sub.h"
#include "ray/rpc/gcs_server/gcs_rpc_server.h"
#include "ray/rpc/worker/core_worker_client.h"
#include "ray/rpc/worker/core_worker_client_pool.h"
#include "ray/util/event.h"
#include "ray/util/thread_checker.h"

namespace ray {
namespace gcs {

// Please keep this in sync with the definition in ray_constants.py.
const std::string kRayInternalNamespacePrefix = "_ray_internal_";

// Please keep these in sync with the definition in dashboard/modules/job/common.py.
const std::string kJobDataKeyPrefix = kRayInternalNamespacePrefix + "job_info_";
inline std::string JobDataKey(const std::string &submission_id) {
  return kJobDataKeyPrefix + submission_id;
}

using JobFinishListenerCallback = rpc::JobInfoHandler::JobFinishListenerCallback;

/// This implementation class of `JobInfoHandler`.
class GcsJobManager : public rpc::JobInfoHandler {
 public:
  explicit GcsJobManager(std::shared_ptr<GcsTableStorage> gcs_table_storage,
                         std::shared_ptr<GcsPublisher> gcs_publisher,
                         RuntimeEnvManager &runtime_env_manager,
                         GcsFunctionManager &function_manager,
                         InternalKVInterface &internal_kv,
                         rpc::CoreWorkerClientFactoryFn client_factory = nullptr)
      : gcs_table_storage_(std::move(gcs_table_storage)),
        gcs_publisher_(std::move(gcs_publisher)),
        runtime_env_manager_(runtime_env_manager),
        function_manager_(function_manager),
        internal_kv_(internal_kv),
        core_worker_clients_(client_factory) {}

  void Initialize(const GcsInitData &gcs_init_data);

  void HandleAddJob(rpc::AddJobRequest request,
                    rpc::AddJobReply *reply,
                    rpc::SendReplyCallback send_reply_callback) override;

  void HandleMarkJobFinished(rpc::MarkJobFinishedRequest request,
                             rpc::MarkJobFinishedReply *reply,
                             rpc::SendReplyCallback send_reply_callback) override;

  void HandleGetAllJobInfo(rpc::GetAllJobInfoRequest request,
                           rpc::GetAllJobInfoReply *reply,
                           rpc::SendReplyCallback send_reply_callback) override;

  void HandleReportJobError(rpc::ReportJobErrorRequest request,
                            rpc::ReportJobErrorReply *reply,
                            rpc::SendReplyCallback send_reply_callback) override;

  void HandleGetNextJobID(rpc::GetNextJobIDRequest request,
                          rpc::GetNextJobIDReply *reply,
                          rpc::SendReplyCallback send_reply_callback) override;

  void AddJobFinishedListener(JobFinishListenerCallback listener) override;

  std::shared_ptr<rpc::JobConfig> GetJobConfig(const JobID &job_id) const;

  /// Handle a node death. This will marks all jobs associated with the
  /// specified node id as finished.
  ///
  /// \param node_id The specified node id.
  void OnNodeDead(const NodeID &node_id);

  void WriteDriverJobExportEvent(rpc::JobTableData job_data) const;

  /// Record metrics.
  /// For job manager, (1) running jobs count gauge and (2) new finished jobs (whether
  /// succeed or fail) will be reported periodically.
  void RecordMetrics();

 private:
  void ClearJobInfos(const rpc::JobTableData &job_data);

  void MarkJobAsFinished(rpc::JobTableData job_table_data,
                         std::function<void(Status)> done_callback);

  // Used to validate invariants for threading; for example, all callbacks are executed on
  // the same thread.
  ThreadChecker thread_checker_;

  // Running Job IDs, used to report metrics.
  absl::flat_hash_set<JobID> running_job_ids_;

  // Number of finished jobs since start of this GCS Server, used to report metrics.
  int64_t finished_jobs_count_ = 0;

  std::shared_ptr<GcsTableStorage> gcs_table_storage_;
  std::shared_ptr<GcsPublisher> gcs_publisher_;

  /// Listeners which monitors the finish of jobs.
  std::vector<JobFinishListenerCallback> job_finished_listeners_;

  /// A cached mapping from job id to job config.
  absl::flat_hash_map<JobID, std::shared_ptr<rpc::JobConfig>> cached_job_configs_;

  ray::RuntimeEnvManager &runtime_env_manager_;
  GcsFunctionManager &function_manager_;
  InternalKVInterface &internal_kv_;

  /// The cached core worker clients which are used to communicate with workers.
  rpc::CoreWorkerClientPool core_worker_clients_;
};

}  // namespace gcs
}  // namespace ray

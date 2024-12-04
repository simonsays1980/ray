// Copyright 2022 The Ray Authors.
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

#include <grpcpp/server.h>
#include <gtest/gtest_prod.h>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "boost/functional/hash.hpp"
#include "ray/common/asio/instrumented_io_context.h"
#include "ray/common/asio/periodical_runner.h"
#include "ray/common/id.h"
#include "ray/common/ray_syncer/common.h"
#include "src/ray/protobuf/ray_syncer.grpc.pb.h"

namespace ray::syncer {

using ray::rpc::syncer::CommandsSyncMessage;
using ray::rpc::syncer::MessageType;
using ray::rpc::syncer::RaySyncMessage;
using ray::rpc::syncer::ResourceViewSyncMessage;

/// The interface for a reporter. Reporter is defined to be a local module which would
/// like to let the other nodes know its state. For example, local cluster resource
/// manager.
struct ReporterInterface {
  /// Interface to get the sync message of the component. It asks the module to take a
  /// snapshot of the current state. Each message is versioned and it should return
  /// std::nullopt if it doesn't have qualified one. The semantics of version depends
  /// on the actual component.
  ///
  /// \param version_after Request message with version after `version_after`. If the
  /// reporter doesn't have the qualified one, just return std::nullopt
  /// \param message_type The message type asked for.
  ///
  /// \return std::nullopt if the reporter doesn't have such component or the current
  /// snapshot of the component is not newer the asked one. Otherwise, return the
  /// actual message.
  virtual std::optional<RaySyncMessage> CreateSyncMessage(
      int64_t version_after, MessageType message_type) const = 0;
  virtual ~ReporterInterface() {}
};

/// The interface for a receiver. Receiver is defined to be a module which would like
/// to get the state of other nodes. For example, cluster resource manager.
struct ReceiverInterface {
  /// Interface to consume a message generated by the other nodes. The module should
  /// read the `sync_message` fields and deserialize it to update its internal state.
  ///
  /// \param message The message received from remote node.
  virtual void ConsumeSyncMessage(std::shared_ptr<const RaySyncMessage> message) = 0;

  virtual ~ReceiverInterface() {}
};

// Forward declaration of internal structures
class NodeState;
class RaySyncerBidiReactor;

/// RaySyncer is an embedding service for component synchronization.
/// All operations in this class needs to be finished GetIOContext()
/// for thread-safety.
/// RaySyncer is the control plane to make sure all connections eventually
/// have the latest view of the cluster components registered.
/// RaySyncer has two components:
///    1. RaySyncerBidiReactor: keeps track of the sending and receiving information
///       and make sure not sending the information the remote node knows.
///    2. NodeState: keeps track of the local status, similar to RaySyncerBidiReactor,
//        but it's for local node.
class RaySyncer {
 public:
  /// Constructor of RaySyncer
  ///
  /// \param io_context The io context for this component.
  /// \param node_id The id of current node.
  RaySyncer(instrumented_io_context &io_context, const std::string &node_id);
  ~RaySyncer();

  /// Connect to a node.
  /// TODO (iycheng): Introduce grpc channel pool and use node_id
  /// for the connection.
  ///
  /// \param node_id The id of the node connect to.
  /// \param channel The gRPC channel.
  void Connect(const std::string &node_id, std::shared_ptr<grpc::Channel> channel);

  void Disconnect(const std::string &node_id);

  /// Get the latest sync message sent from a specific node.
  ///
  /// \param node_id The node id where the message comes from.
  /// \param message_type The message type of the component.
  ///
  /// \return The latest sync message sent from the node. If the node doesn't
  /// have one, nullptr will be returned.
  std::shared_ptr<const RaySyncMessage> GetSyncMessage(const std::string &node_id,
                                                       MessageType message_type) const;

  /// Register the components to the syncer module. Syncer will make sure eventually
  /// it'll have a global view of the cluster.
  ///
  ///
  /// \param message_type The message type of the component.
  /// \param reporter The local component to be broadcasted.
  /// \param receiver The consumer of the sync message sent by the other nodes in the
  /// cluster.
  /// \param pull_from_reporter_interval_ms The frequence to pull a message. 0 means
  /// never pull a message in syncer.
  /// from reporter and push it to sending queue.
  void Register(MessageType message_type,
                const ReporterInterface *reporter,
                ReceiverInterface *receiver,
                int64_t pull_from_reporter_interval_ms = 100);

  /// Get the current node id.
  const std::string &GetLocalNodeID() const { return local_node_id_; }

  /// Request trigger a broadcasting for a specific component immediately instead of
  /// waiting for ray syncer to poll the message.
  ///
  /// \param message_type The component to check.
  /// \return true if a message is generated. If the component doesn't have a new
  /// version of message, false will be returned.
  bool OnDemandBroadcasting(MessageType message_type);

  /// WARNING: DON'T USE THIS METHOD. It breaks the abstraction of the syncer.
  /// Instead, register the component to the syncer and call
  /// OnDemandBroadcasting.
  ///
  /// Request trigger a broadcasting for a constructed message immediately instead of
  /// waiting for ray syncer to poll the message.
  ///
  /// \param message The message to be broadcasted.
  void BroadcastRaySyncMessage(std::shared_ptr<const RaySyncMessage> message);

  std::vector<std::string> GetAllConnectedNodeIDs() const;

 private:
  void Connect(RaySyncerBidiReactor *connection);

  std::shared_ptr<bool> stopped_;

  /// Get the io_context used by RaySyncer.
  instrumented_io_context &GetIOContext() { return io_context_; }

  /// Function to broadcast the messages to other nodes.
  /// A message will be sent to a node if that node doesn't have this message.
  /// The message can be generated by local reporter or received by the other node.
  ///
  /// \param message The message to be broadcasted.
  void BroadcastMessage(std::shared_ptr<const RaySyncMessage> message);

  /// io_context for this thread
  instrumented_io_context &io_context_;

  /// The current node id.
  const std::string local_node_id_;

  /// Manage connections. Here the key is the NodeID in binary form.
  absl::flat_hash_map<std::string, RaySyncerBidiReactor *> sync_reactors_;

  /// The local node state
  std::unique_ptr<NodeState> node_state_;

  /// Timer is used to do broadcasting.
  ray::PeriodicalRunner timer_;

  friend class RaySyncerService;
  /// Test purpose
  friend struct SyncerServerTest;
  FRIEND_TEST(SyncerTest, Broadcast);
  FRIEND_TEST(SyncerTest, Reconnect);
  FRIEND_TEST(SyncerTest, Test1To1);
  FRIEND_TEST(SyncerTest, Test1ToN);
  FRIEND_TEST(SyncerTest, TestMToN);
  FRIEND_TEST(SyncerTest, Reconnect);
};

/// RaySyncerService is a service to take care of resource synchronization
/// related operations.
/// Right now only raylet needs to setup this service. But in the future,
/// we can use this to construct more complicated resource reporting algorithm,
/// like tree-based one.
class RaySyncerService : public ray::rpc::syncer::RaySyncer::CallbackService {
 public:
  RaySyncerService(RaySyncer &syncer) : syncer_(syncer) {}

  ~RaySyncerService();

  grpc::ServerBidiReactor<RaySyncMessage, RaySyncMessage> *StartSync(
      grpc::CallbackServerContext *context) override;

 private:
  // The ray syncer this RPC wrappers of.
  RaySyncer &syncer_;
};

}  // namespace ray::syncer

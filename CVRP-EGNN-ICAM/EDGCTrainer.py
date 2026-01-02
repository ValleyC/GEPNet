"""
EDGC Trainer for CVRP.

One-shot clustering approach replacing autoregressive partition.

Key differences from CVRPTrainerPartition:
1. O(N) one-shot clustering instead of O(N^2) autoregressive
2. Combined loss: InfoNCE + KL + Capacity + REINFORCE
3. Cluster-to-route conversion with capacity splitting
"""

import torch
import numpy as np
from logging import getLogger
from torch_geometric.data import Data
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.distributions import Categorical

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from ClusterPartitionModel import ClusterPartitionModel
from ClusterLosses import EDGCLoss, compute_cluster_statistics
from utils.utils import get_result_folder, AverageMeter, LogData, TimeEstimator, util_print_log_array


class EDGCTrainer:
    """
    EDGC Trainer for CVRP.

    Uses one-shot clustering for partition instead of autoregressive sampling.
    """

    def __init__(self, env_params, model_params, model_p_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.model_p_params = model_p_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # CUDA setup
        USE_CUDA = trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Cluster Partition Model (EDGC)
        self.model_p = ClusterPartitionModel(
            units=model_p_params['embedding_dim'],
            node_feats=2,  # (demand, r) - E(2)-invariant
            edge_feats=2,
            depth=model_p_params['depth'],
            projection_dim=model_p_params.get('projection_dim', 128),
            max_clusters=model_p_params.get('max_clusters', 50)
        ).to(self.device)

        # Sub-problem solver (TSP model)
        self.model_t = Model(**model_params)
        self.env = Env(**env_params)

        # EDGC Loss
        self.edgc_loss = EDGCLoss(
            w_infonce=trainer_params.get('w_infonce', 1.0),
            w_kl=trainer_params.get('w_kl', 0.1),
            w_capacity=trainer_params.get('w_capacity', 10.0),
            w_balance=trainer_params.get('w_balance', 0.01),
            w_reinforce=trainer_params.get('w_reinforce', 1.0),
            temperature=trainer_params.get('temperature', 0.5)
        )

        # Optimizers
        self.optimizer_p = Optimizer(self.model_p.parameters(), **optimizer_params['optimizer_p'])
        self.optimizer_t = Optimizer(self.model_t.parameters(), **optimizer_params['optimizer'])
        self.scheduler_p = Scheduler(self.optimizer_p, **optimizer_params['scheduler'])
        self.scheduler_t = Scheduler(self.optimizer_t, **optimizer_params['scheduler'])

        self.start_epoch = 1
        self._load_checkpoints()
        self.time_estimator = TimeEstimator()

    def _load_checkpoints(self):
        """Load pre-trained models if specified."""
        model_load = self.trainer_params['model_load']

        if model_load['t_enable']:
            checkpoint_fullname = '{t_path}/checkpoint-tsp-{t_epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model_t.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['t_epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer_t.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_t.last_epoch = model_load['t_epoch'] - 1
            self.logger.info('Loaded TSP Model!')

        if model_load['p_enable']:
            checkpoint_fullname = '{p_path}/checkpoint-partition-{p_epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model_p.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['p_epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer_p.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_p.last_epoch = model_load['p_epoch'] - 1
            self.logger.info('Loaded EDGC Partition Model!')

    def run(self):
        """Main training loop."""
        self.time_estimator.reset(self.start_epoch)

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=' * 60)

            self.scheduler_p.step()
            self.scheduler_t.step()

            train_score, train_loss, cluster_stats = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            self.logger.info("Cluster Stats: {}".format(cluster_stats))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            if all_done or (epoch % model_save_interval) == 0:
                self._save_checkpoints(epoch)

            if all_done:
                self.logger.info(" *** Training Done *** ")
                util_print_log_array(self.logger, self.result_log)

    def _save_checkpoints(self, epoch):
        """Save model checkpoints."""
        self.logger.info("Saving checkpoints...")

        checkpoint_dict_t = {
            'epoch': epoch,
            'model_state_dict': self.model_t.state_dict(),
            'optimizer_state_dict': self.optimizer_t.state_dict(),
            'scheduler_state_dict': self.scheduler_t.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }
        torch.save(checkpoint_dict_t, f'{self.result_folder}/checkpoint-tsp-{epoch}.pt')

        checkpoint_dict_p = {
            'epoch': epoch,
            'model_state_dict': self.model_p.state_dict(),
            'optimizer_state_dict': self.optimizer_p.state_dict(),
            'scheduler_state_dict': self.scheduler_p.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }
        torch.save(checkpoint_dict_p, f'{self.result_folder}/checkpoint-partition-{epoch}.pt')

    def _train_one_epoch(self, epoch):
        """Train for one epoch."""
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        cluster_stats_AM = {}

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, stats = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            # Update cluster statistics
            for k, v in stats.items():
                if k not in cluster_stats_AM:
                    cluster_stats_AM[k] = AverageMeter()
                cluster_stats_AM[k].update(v, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                        .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                score_AM.avg, loss_AM.avg))

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
            .format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))

        final_stats = {k: v.avg for k, v in cluster_stats_AM.items()}
        return score_AM.avg, loss_AM.avg, final_stats

    def _train_one_batch(self, batch_size):
        """Train on one batch using one-shot clustering."""
        self.model_t.train()
        self.model_p.train()

        # Generate problems
        self.env.load_raw_problems(batch_size)
        pyg_data = self.gen_pyg_data(self.env.raw_depot_node_xy, self.env.raw_depot_node_demand)

        # Get demands (excluding depot)
        demands = self.env.raw_depot_node_demand[0, 1:]  # (n_nodes-1,)

        # Estimate number of clusters based on total demand
        total_demand = demands.sum()
        n_clusters = max(int(torch.ceil(total_demand).item()) + 1, 2)
        n_clusters = min(n_clusters, 50)

        # One-shot clustering (O(N) instead of O(N^2))
        q, z1, z2, k_logits, distances = self.model_p(
            pyg_data,
            n_clusters=n_clusters,
            is_train=True,
            sigma=0.01
        )

        # Get hard cluster assignments (for solver) - use multiple samples for POMO-like exploration
        # Sample multiple times and keep the best
        sample_size = self.env.sample_size
        all_labels = []
        all_log_probs = []

        for _ in range(sample_size):
            labels, log_probs = self.model_p.get_hard_assignment(q[1:], sample=True)
            all_labels.append(labels)
            all_log_probs.append(log_probs)

        all_labels = torch.stack(all_labels, dim=0)  # (sample_size, n_nodes)
        all_log_probs = torch.stack(all_log_probs, dim=0)  # (sample_size, n_nodes)

        # Compute cluster statistics (using first sample)
        stats = compute_cluster_statistics(q[1:], demands)

        # Convert each sampled clustering to solution and solve
        solution, solution_flag = self._clusters_to_solution(all_labels, demands)

        # Solve sub-problems using the conquering stage
        merge_reward, loss_t = self._solve_with_conquering(solution, solution_flag, batch_size)

        # REINFORCE for partition
        log_probs_sum = all_log_probs.sum(dim=1)  # (sample_size,)
        advantage = merge_reward - merge_reward.mean()
        loss_partition = -(advantage.detach() * log_probs_sum).mean()

        # Compute EDGC clustering losses
        total_loss, loss_dict = self.edgc_loss(
            z1[1:], z2[1:], q[1:],
            demands=demands,
            capacity=1.0,
            log_probs=None,  # Already handled above
            rewards=None,
            baseline=None
        )

        # Combine losses
        total_loss = total_loss + loss_partition + loss_t

        # Backward pass
        self.model_p.zero_grad()
        self.model_t.zero_grad()
        total_loss.backward()
        self.optimizer_p.step()
        self.optimizer_t.step()

        max_reward = merge_reward.max()
        score = -max_reward.item()
        return score, total_loss.item(), stats

    def _clusters_to_solution(self, all_labels, demands, capacity=1.0):
        """
        Convert cluster labels to solution format compatible with conquering stage.

        Args:
            all_labels: (sample_size, n_nodes) cluster assignments
            demands: (n_nodes,) node demands
            capacity: vehicle capacity

        Returns:
            solution: (sample_size, n_nodes) node visit order
            solution_flag: (sample_size, n_nodes) route end markers
        """
        sample_size = all_labels.shape[0]
        n_nodes = all_labels.shape[1]
        device = all_labels.device

        solution = torch.zeros((sample_size, n_nodes), dtype=torch.long, device=device)
        solution_flag = torch.zeros((sample_size, n_nodes), dtype=torch.long, device=device)

        coords = self.env.raw_depot_node_xy[0]  # (n_nodes+1, 2)
        depot_coord = coords[0]

        for s in range(sample_size):
            labels = all_labels[s]
            unique_clusters = torch.unique(labels)

            current_pos = 0
            for cluster_id in unique_clusters:
                mask = labels == cluster_id
                cluster_nodes = torch.where(mask)[0] + 1  # +1 for depot offset (1-indexed)
                cluster_demands = demands[mask]

                # Sort by angle from depot
                cluster_coords = coords[cluster_nodes]
                rel_coords = cluster_coords - depot_coord
                angles = torch.atan2(rel_coords[:, 1], rel_coords[:, 0])
                sorted_indices = torch.argsort(angles)

                # Split into routes based on capacity
                current_demand = 0.0
                route_start = current_pos

                for idx in sorted_indices:
                    node = cluster_nodes[idx]
                    demand = cluster_demands[idx].item()

                    if current_demand + demand > capacity + 1e-6:
                        # Mark end of route
                        if current_pos > route_start:
                            solution_flag[s, current_pos - 1] = 1
                        route_start = current_pos
                        current_demand = demand
                    else:
                        current_demand += demand

                    solution[s, current_pos] = node
                    current_pos += 1

                # Mark end of cluster's last route
                if current_pos > route_start:
                    solution_flag[s, current_pos - 1] = 1

        # Final flag
        solution_flag[:, -1] = 1

        return solution, solution_flag

    def _solve_with_conquering(self, solution, solution_flag, batch_size):
        """
        Apply conquering stage (DCR) to optimize sub-problems.

        Similar to original CVRPTrainerPartition but adapted for clustering output.
        """
        loss_t_total = torch.tensor(0.0, device=self.device)

        # Route ranking by angle
        solution, solution_flag = self.route_ranking(
            self.env.raw_depot_node_xy, solution, solution_flag
        )

        # Reshape into sub-problems of size problem_size
        n_tsps_per_route = solution.view(solution.size(0), -1, self.env.problem_size)
        n_tsps_per_route_flag = solution_flag.view(solution.size(0), -1, self.env.problem_size)

        # Get demands for each sub-problem
        demand_per_route = self.env.raw_depot_node_demand[:, None, :].repeat(
            solution.size(0), n_tsps_per_route.size(1), 1
        ).gather(-1, n_tsps_per_route)

        # Capacity calculation
        capacity_now = torch.ones((demand_per_route.size(0), demand_per_route.size(1)), device=self.device)
        tag = (n_tsps_per_route_flag * (n_tsps_per_route.size(-1) - torch.arange(n_tsps_per_route.size(-1), device=self.device))).max(-1)[1].unsqueeze(-1)
        capacity_now -= torch.cumsum(demand_per_route, dim=-1).gather(-1, tag).squeeze()

        capacity_end = torch.ones((demand_per_route.size(0), demand_per_route.size(1)), device=self.device)
        tag = (n_tsps_per_route_flag * torch.arange(n_tsps_per_route.size(-1), device=self.device)).max(-1)[1].unsqueeze(-1)
        capacity_end -= torch.cumsum(demand_per_route, dim=-1)[:, :, -1] - torch.cumsum(demand_per_route, dim=-1).gather(-1, tag).squeeze()

        # Create TSP instances
        tsp_insts = self.env.raw_problems[:, None, :].repeat(
            solution.size(0), n_tsps_per_route.size(1), 1, 1
        ).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 3))
        customer_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))

        # Add depot
        tsp_insts_now = torch.cat((
            self.env.raw_problems[:, 0, :].unsqueeze(0).repeat(customer_insts_now.size(0), 1, 1),
            customer_insts_now
        ), dim=1)

        # Initial solution
        solution_now = torch.arange(1, tsp_insts_now.size(-2), device=self.device)[None, :].expand((tsp_insts_now.size(0), -1))[:, None, :]
        reward_now = self.env.cal_open_length(
            tsp_insts_now[:, :, [0, 1]],
            solution_now,
            n_tsps_per_route_flag.view(-1, tsp_insts_now.size(-2) - 1)[:, None, :]
        )

        # Capacity pairing
        capacity_pair2 = capacity_end.clone().view(-1, 1)
        capacity_pair1 = capacity_now.clone().roll(dims=1, shifts=-1).view(-1, 1)
        capacity_pair = torch.cat((capacity_pair1, capacity_pair2), dim=-1)
        capacity_pair[:, 0][(capacity_pair[:, 1] == 1.)] = 0.
        tag = ((capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] > 0.5)).clone()
        capacity_pair[:, 1][tag] = 0.5
        capacity_pair[:, 0][tag] = 0.5
        capacity_pair[:, 0][(capacity_pair[:, 0] > 0.5) & (capacity_pair[:, 1] <= 0.5)] = 1 - capacity_pair[:, 1][(capacity_pair[:, 0] > 0.5) & (capacity_pair[:, 1] <= 0.5)]
        capacity_pair[:, 1][(capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5)] = 1 - capacity_pair[:, 0][(capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5)]

        capacity_head = capacity_pair[:, 1].clone().view(self.env.sample_size, -1).roll(dims=1, shifts=1).view(-1, 1)
        capacity_tail = capacity_pair[:, 0].clone().view(-1, 1)
        new_batch_size = tsp_insts_now.size(0)

        # Load sub-problems
        self.env.load_problems(
            new_batch_size,
            tsp_insts_now[:, 0:1, :2],
            tsp_insts_now[:, 1:, :2],
            tsp_insts_now[:, 1:, -1],
            n_tsps_per_route_flag[:, :, -1].clone().view(-1)
        )
        reset_state, _, _ = self.env.reset(capacity_head, capacity_tail)
        self.model_t.pre_forward(reset_state)

        # POMO rollout
        prob_list = torch.zeros(size=(new_batch_size, self.env.pomo_size, 0), device=self.device)
        state, reward, done = self.env.pre_step()

        while not done:
            cur_dist = self.env.get_local_feature()
            selected_t, prob = self.model_t(state, cur_dist, n_tsps_per_route_flag[:, :, -1].clone().view(-1))
            state, reward, done = self.env.step(selected_t)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        new_solution = torch.cat((state.solution_list.unsqueeze(-1), state.solution_flag.unsqueeze(-1)), dim=-1)
        reward = -1 * self.env.cal_length(tsp_insts_now[:, :, [0, 1]], new_solution[:, :, :, 0], new_solution[:, :, :, 1])

        # Sub-solver loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        loss = (-advantage * prob_list.log().sum(dim=2)).mean()
        loss_t_total = loss_t_total + loss

        # Update solution with best POMO results
        reward_eval = self.env.cal_length(tsp_insts_now[:, :, [0, 1]], new_solution[:, :, :, 0], new_solution[:, :, :, 1])
        tag = reward_eval.view(batch_size, self.env.sample_size, -1, self.env.pomo_size).min(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
        tag_solution = state.solution_list.view(batch_size, self.env.sample_size, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze()
        tag_solution_flag = state.solution_flag.view(batch_size, self.env.sample_size, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze()

        r = (reward_eval.min(1)[0] > reward_now.squeeze()).view(self.env.sample_size, -1, 1).expand((-1, -1, tsp_insts_now.size(-2) - 1))
        tag_solution[r] = solution_now.view(self.env.sample_size, -1, tsp_insts_now.size(-2) - 1)[r]
        tag_solution_flag[r] = n_tsps_per_route_flag[r]

        solution = n_tsps_per_route.gather(-1, tag_solution - 1).view(solution.size(0), -1)
        solution_flag = tag_solution_flag.view(solution.size(0), -1)

        # Compute final reward
        merge_reward = -1 * self.env.cal_length_total(
            self.env.raw_problems[:, :, [0, 1]], solution, solution_flag
        ).squeeze(0)

        return merge_reward, loss_t_total

    def route_ranking(self, problem, solution, solution_flag):
        """Rank routes by angle for better partitioning."""
        roll = ((solution_flag * torch.arange(solution.size(-1), device=solution.device)[None, :]).max(-1)[1] + 1) % solution.size(-1)
        roll_init = solution.size(-1) - roll[:, None]
        roll_diff = (torch.arange(solution.size(-1), device=solution.device)[None, :].expand_as(solution) + roll[:, None]) % solution.size(-1)
        now_solution = solution.gather(1, roll_diff)
        now_solution_flag = solution_flag.gather(1, roll_diff)
        solution = now_solution.clone()
        solution_flag = now_solution_flag.clone()

        vector = problem - problem[:, 0, :][:, None, :]
        vector_rank = vector.repeat(solution.size(0), 1, 1).gather(1, solution.unsqueeze(-1).expand(-1, -1, 2))
        solution_start = torch.cummax(solution_flag.roll(dims=1, shifts=1) * torch.arange(solution.size(-1), device=solution.device)[None, :], dim=-1)[0]
        solution_end = solution.size(-1) - 1 - torch.flip(torch.cummax(torch.flip(solution_flag, dims=[1]) * torch.arange(solution.size(-1), device=solution.device)[None, :], dim=-1)[0], dims=[1])
        num_vector2 = solution_end - solution_start + 1

        cum_vr = torch.cumsum(vector_rank.clone(), dim=1)
        sum_vector2 = cum_vr.clone().gather(1, solution_end.unsqueeze(-1).expand_as(vector_rank)) - \
                      cum_vr.clone().gather(1, solution_start.unsqueeze(-1).expand_as(vector_rank)) + \
                      vector_rank.clone().gather(1, solution_start.unsqueeze(-1).expand_as(vector_rank))

        vector_angle = torch.atan2(sum_vector2[:, :, 1] / num_vector2, sum_vector2[:, :, 0] / num_vector2)
        total_indi = vector_angle
        total_rank = np.argsort(total_indi.cpu().numpy(), kind='stable')
        total_rank = torch.from_numpy(total_rank).to(solution.device)

        roll = total_rank.min(-1)[1]
        roll_diff = (torch.arange(solution.size(-1), device=solution.device)[None, :].expand_as(solution) + roll[:, None]) % solution.size(-1)
        now_rank = total_rank.gather(1, roll_diff)

        solution_rank = solution.gather(1, now_rank)
        solution_flag_rank = solution_flag.gather(1, now_rank)

        roll_diff = (torch.arange(solution.size(-1), device=solution.device)[None, :].expand_as(solution) + roll_init) % solution.size(-1)
        solution_rank = solution_rank.gather(1, roll_diff)
        solution_flag_rank = solution_flag_rank.gather(1, roll_diff)

        return solution_rank, solution_flag_rank

    def gen_pyg_data(self, coors, demand, k_sparse=100):
        """Generate PyG data with E(2)-invariant features."""
        n_nodes = demand.size(1)
        norm_demand = demand

        shift_coors = coors - coors[:, 0:1, :]
        _x, _y = shift_coors[:, :, 0], shift_coors[:, :, 1]
        r = torch.sqrt(_x ** 2 + _y ** 2)

        x = torch.stack((norm_demand, r)).permute(1, 2, 0)
        cos_mat = self.gen_cos_sim_matrix(shift_coors)

        topk_values, topk_indices = torch.topk(cos_mat, k=min(k_sparse, n_nodes), dim=2, largest=True)

        edge_index = torch.cat((
            torch.repeat_interleave(torch.arange(n_nodes, device=coors.device), repeats=min(k_sparse, n_nodes))[None, :],
            topk_indices.view(1, -1)
        ), dim=0)

        edge_attr1 = topk_values.reshape(1, -1, 1)
        edge_attr2 = cos_mat[0, edge_index[0], edge_index[1]].reshape(1, -1, 1)
        edge_attr = torch.cat((edge_attr1, edge_attr2), dim=2)

        pyg_data = Data(
            x=x[0],
            edge_index=edge_index,
            edge_attr=edge_attr[0],
            pos=coors[0]
        )
        return pyg_data

    def gen_cos_sim_matrix(self, shift_coors):
        """Compute cosine similarity matrix."""
        dot_products = torch.bmm(shift_coors, shift_coors.transpose(1, 2))
        magnitudes = torch.sqrt(torch.sum(shift_coors ** 2, dim=-1)).unsqueeze(-1)
        magnitude_matrix = torch.bmm(magnitudes, magnitudes.transpose(1, 2)) + 1e-10
        cosine_similarity_matrix = dot_products / magnitude_matrix
        return cosine_similarity_matrix

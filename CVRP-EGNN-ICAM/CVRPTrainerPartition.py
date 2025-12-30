"""
GEPNet Trainer for CVRP.

Key innovation: No coordinate_transformation needed!
EGNN produces invariant embeddings directly.
"""

import torch
from logging import getLogger
from torch_geometric.data import Data
import numpy as np
from torch.distributions import Categorical
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from PartitionModel import PartitionModel
from utils.utils import get_result_folder, AverageMeter, LogData, TimeEstimator, util_print_log_array


class CVRPPartitionTrainer:
    """
    GEPNet Trainer for CVRP.

    Key differences from UDC's trainer:
    1. No coordinate_transformation in conquering stage
    2. EGNN-based partition model
    3. Invariant node features by construction
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

        # Models
        self.model_p = PartitionModel(
            units=model_p_params['embedding_dim'],
            feats=2,  # (demand, r) - NO theta for E(2)-equivariance!
            k_sparse=100,
            edge_feats=2,
            depth=model_p_params['depth'],
            use_egnn=model_p_params.get('use_egnn', True)
        ).to(self.device)

        self.model_t = Model(**model_params)
        self.env = Env(**env_params)

        # Optimizers
        self.optimizer_p = Optimizer(self.model_p.parameters(), **optimizer_params['optimizer_p'])
        self.optimizer_t = Optimizer(self.model_t.parameters(), **optimizer_params['optimizer'])
        self.scheduler_p = Scheduler(self.optimizer_p, **optimizer_params['scheduler'])
        self.scheduler_t = Scheduler(self.optimizer_t, **optimizer_params['scheduler'])

        # Load checkpoints
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
            self.logger.info('Loaded Partition Model!')

    def run(self):
        """Main training loop."""
        self.time_estimator.reset(self.start_epoch)

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=' * 60)

            self.scheduler_p.step()
            self.scheduler_t.step()

            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

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
        loss_P = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, loss_partition = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            loss_P.update(loss_partition, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  Loss_P: {:.4f}'
                        .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                score_AM.avg, loss_AM.avg, loss_P.avg))

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Loss_P: {:.4f}'
            .format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg, loss_P.avg))

        return score_AM.avg, loss_AM.avg

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

    def _train_one_batch(self, batch_size):
        """Train on one batch."""
        self.model_t.train()
        self.model_p.train()

        # Generate problems
        self.env.load_raw_problems(batch_size)
        pyg_data = self.gen_pyg_data(self.env.raw_depot_node_xy, self.env.raw_depot_node_demand)

        # Initialize partition
        logp = torch.zeros(self.env.sample_size, dtype=torch.float32, device=self.device)
        index = torch.zeros(self.env.sample_size, dtype=torch.long, device=self.device)
        vehicle_count = torch.zeros((self.env.sample_size,), device=self.device)
        demand_count = torch.zeros((self.env.sample_size,), device=self.device)
        visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1))
        solution_raw = index[:, None]
        visited = visited.scatter(-1, solution_raw[:, 0:1], 1)
        selected = solution_raw

        # Pre-compute embeddings (EGNN - no coordinate normalization!)
        self.model_p.pre(pyg_data)

        max_vehicle = (self.env.raw_depot_node_demand.sum().ceil() + 1).item()
        total_demand = self.env.raw_depot_node_demand.sum()
        remaining_demand = total_demand.clone()

        solution = torch.zeros((self.env.sample_size, self.env.raw_problems.size(1) - 1), dtype=torch.long, device=self.device)
        solution[:, 0] = solution_raw[:, 0]
        step = 0
        solution_flag = torch.zeros((self.env.sample_size, self.env.raw_problems.size(1) - 1), dtype=torch.long, device=self.device)
        node_count = -1 * torch.ones((self.env.sample_size, 1), dtype=torch.long, device=self.device)
        capacity = torch.ones_like(index)[:, None].float()
        capacity -= self.env.raw_depot_node_demand.expand((self.env.sample_size, -1)).gather(-1, selected)

        # Autoregressive partition
        for i in range(self.env.raw_problem_size // self.env.problem_size):
            node_emb, heatmap = self.model_p(solution, selected, visited)
            heatmap = heatmap / (heatmap.min() + 1e-5)
            heatmap = self.model_p.reshape(pyg_data, heatmap) + 1e-5

            while ((visited.sum(-1) - i * self.env.problem_size) < self.env.problem_size).any():
                step += 1
                capacity_mask = (self.env.raw_depot_node_demand > capacity + 1e-5).long().squeeze()
                row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).clone().squeeze(1) * (1 - visited).clone() * (1 - capacity_mask).clone()
                row[:, 0][selected[:, 0] == 0] = 0
                row[:, 0][remaining_demand > (max_vehicle - vehicle_count)] = 0
                row[:, 0][row[:, 1:].sum(-1) < 1e-8] = 1

                dist = Categorical(row)
                item = dist.sample()
                log_prob = dist.log_prob(item)

                selected = item[:, None]
                logp += log_prob
                demand_count += self.env.raw_depot_node_demand.expand(self.env.sample_size, -1).gather(1, selected).squeeze()
                remaining_demand = total_demand - demand_count
                visited = visited.scatter(-1, selected, 1)
                visited[:, 0] = 0
                capacity -= self.env.raw_depot_node_demand.expand((self.env.sample_size, -1)).gather(-1, selected)
                capacity[selected == 0] = 1
                vehicle_count[item == 0] += 1

                if step > 1:
                    solution_flag = solution_flag.scatter_add(dim=-1, index=node_count, src=(selected == 0).long())
                node_count[selected != 0] += 1
                solution = solution.scatter_add(dim=-1, index=node_count, src=selected)

        solution_flag[:, -1] = 1

        # DCR-enabled conquering stage
        loss_t_total = 0
        for i in range(2):
            solution, solution_flag = self.route_ranking(self.env.raw_depot_node_xy, solution, solution_flag)
            roll = self.env.problem_size // 2
            solution = solution.roll(dims=1, shifts=roll)
            solution_flag = solution_flag.roll(dims=1, shifts=roll)

            n_tsps_per_route = solution.view(solution.size(0), -1, self.env.problem_size)
            n_tsps_per_route_flag = solution_flag.view(solution.size(0), -1, self.env.problem_size)
            demand_per_route = self.env.raw_depot_node_demand[:, None, :].repeat(solution.size(0), n_tsps_per_route.size(1), 1).gather(-1, n_tsps_per_route)

            # Capacity calculation
            capacity_now = torch.ones((demand_per_route.size(0), demand_per_route.size(1)), device=self.device)
            tag = (n_tsps_per_route_flag * (n_tsps_per_route.size(-1) - torch.arange(n_tsps_per_route.size(-1), device=self.device))).max(-1)[1].unsqueeze(-1)
            capacity_now -= torch.cumsum(demand_per_route, dim=-1).gather(-1, tag).squeeze()

            capacity_end = torch.ones((demand_per_route.size(0), demand_per_route.size(1)), device=self.device)
            tag = (n_tsps_per_route_flag * torch.arange(n_tsps_per_route.size(-1), device=self.device)).max(-1)[1].unsqueeze(-1)
            capacity_end -= torch.cumsum(demand_per_route, dim=-1)[:, :, -1] - torch.cumsum(demand_per_route, dim=-1).gather(-1, tag).squeeze()

            tsp_insts = self.env.raw_problems[:, None, :].repeat(solution.size(0), n_tsps_per_route.size(1), 1, 1).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 3))
            customer_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))

            # GEPNet: No coordinate transformation needed!
            tsp_insts_now = torch.cat((self.env.raw_problems[:, 0, :].unsqueeze(0).repeat(customer_insts_now.size(0), 1, 1), customer_insts_now), dim=1)
            solution_now = torch.arange(1, tsp_insts_now.size(-2), device=self.device)[None, :].expand((tsp_insts_now.size(0), -1))[:, None, :]
            reward_now = self.env.cal_open_length(tsp_insts_now[:, :, [0, 1]], solution_now, n_tsps_per_route_flag.view(-1, tsp_insts_now.size(-2) - 1)[:, None, :])

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

            # Load sub-problems - NO coordinate transformation!
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

            self.model_t.zero_grad()
            loss.backward()
            self.optimizer_t.step()
            loss_t_total += loss.item()

            # Update solution with best POMO results
            reward = self.env.cal_length(tsp_insts_now[:, :, [0, 1]], new_solution[:, :, :, 0], new_solution[:, :, :, 1])
            tag = reward.view(batch_size, self.env.sample_size, -1, self.env.pomo_size).min(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
            tag_solution = state.solution_list.view(batch_size, self.env.sample_size, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze()
            tag_solution_flag = state.solution_flag.view(batch_size, self.env.sample_size, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze()

            r = (reward.min(1)[0] > reward_now.squeeze()).view(self.env.sample_size, -1, 1).expand((-1, -1, tsp_insts_now.size(-2) - 1))
            tag_solution[r] = solution_now.view(self.env.sample_size, -1, tsp_insts_now.size(-2) - 1)[r]
            tag_solution_flag[r] = n_tsps_per_route_flag[r]

            solution = n_tsps_per_route.gather(-1, tag_solution - 1).view(solution.size(0), -1)
            solution_flag = tag_solution_flag.view(solution.size(0), -1)

        # Partition loss (REINFORCE)
        merge_reward = -1 * self.env.cal_length_total(self.env.raw_problems[:, :, [0, 1]], solution, solution_flag)
        advantage2 = merge_reward - merge_reward.float().mean(dim=1, keepdims=True)
        loss_partition = (-advantage2 * logp).mean()

        max_pomo_reward, _ = merge_reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model_p.zero_grad()
        loss_partition.backward()
        self.optimizer_p.step()

        return score_mean.item(), loss_t_total / 2, loss_partition.item()

    def gen_pyg_data(self, coors, demand, k_sparse=100):
        """
        Generate PyG data with E(2)-invariant features.

        Node features: (demand, r) - demand and distance from depot
        Edge features: (cosine_similarity, distance)

        Note: theta (polar angle) is NOT used because it breaks E(2)-equivariance!
        Under rotation, theta changes, which would produce different outputs.
        """
        bs = demand.size(0)
        n_nodes = demand.size(1)
        norm_demand = demand

        # Shift coordinates relative to depot
        shift_coors = coors - coors[:, 0:1, :]
        _x, _y = shift_coors[:, :, 0], shift_coors[:, :, 1]

        # Distance from depot (E(2)-invariant!)
        r = torch.sqrt(_x ** 2 + _y ** 2)
        # NOTE: theta = torch.atan2(_y, _x) is NOT used - breaks equivariance!

        # Node features: (demand, r) - both are E(2)-invariant
        x = torch.stack((norm_demand, r)).permute(1, 2, 0)

        # Cosine similarity matrix
        cos_mat = self.gen_cos_sim_matrix(shift_coors)

        # Sparse edges
        topk_values, topk_indices = torch.topk(cos_mat, k=k_sparse, dim=2, largest=True)

        edge_index = torch.cat((
            torch.repeat_interleave(torch.arange(n_nodes, device=coors.device), repeats=k_sparse)[None, :].repeat(bs, 1)[:, None, :],
            topk_indices.view(bs, -1)[:, None, :]
        ), dim=1)

        idx = torch.arange(bs, device=coors.device)[:, None, None].repeat(1, n_nodes, k_sparse)
        edge_attr1 = topk_values.reshape(bs, -1, 1)
        edge_attr2 = cos_mat[idx.view(bs, -1), edge_index[:, 0], edge_index[:, 1]].reshape(bs, k_sparse * n_nodes, 1)
        edge_attr = torch.cat((edge_attr1, edge_attr2), dim=2)

        # Create PyG data
        pyg_data = Data(
            x=x[0],
            edge_index=edge_index[0],
            edge_attr=edge_attr[0],
            pos=coors[0]  # Original coordinates for EGNN
        )
        return pyg_data

    def gen_cos_sim_matrix(self, shift_coors):
        """Compute cosine similarity matrix."""
        dot_products = torch.bmm(shift_coors, shift_coors.transpose(1, 2))
        magnitudes = torch.sqrt(torch.sum(shift_coors ** 2, dim=-1)).unsqueeze(-1)
        magnitude_matrix = torch.bmm(magnitudes, magnitudes.transpose(1, 2)) + 1e-10
        cosine_similarity_matrix = dot_products / magnitude_matrix
        return cosine_similarity_matrix

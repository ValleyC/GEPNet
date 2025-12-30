"""
GEPNet Trainer for TSP.

Key innovation: No coordinate_transformation needed!
EGNN produces invariant embeddings directly.
"""

import torch
from logging import getLogger
from torch_geometric.data import Data
from torch.distributions import Categorical
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from PartitionModel import PartitionModel
from utils.utils import get_result_folder, AverageMeter, LogData, TimeEstimator, util_print_log_array


class TSPTrainerPartition:
    """
    GEPNet Trainer for TSP.

    Key differences from UDC:
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
            feats=1,  # Constant feature (1) - E(2)-invariant! Coordinates passed separately.
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
        """Load pre-trained models."""
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

            # Curriculum: start smaller
            if epoch < 50:
                self.env.problem_size_high = self.env.problem_size_low
                self.env.sample_size = self.env.fs_sample_size
            else:
                self.env.problem_size_high = self.env_params['problem_size_high']
                self.env.sample_size = self.env_params['sample_size']

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
        """Save checkpoints."""
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
        """Train one epoch."""
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

    def _train_one_batch(self, batch_size):
        """Train one batch."""
        self.model_t.train()
        self.model_p.train()

        # Generate problems
        self.env.load_raw_problems(batch_size)
        pyg_data = self._gen_pyg_data(self.env.raw_problems)

        # Initialize partition
        index = torch.randint(0, self.env.raw_problems.size(1), [self.env.sample_size], device=self.device)
        logp = torch.zeros(self.env.sample_size, dtype=torch.float32, device=self.device)
        visited = torch.zeros(self.env.sample_size, self.env.raw_problems.size(1), device=self.device)
        solution = index[:, None]
        visited = visited.scatter(-1, solution[:, 0:1], 1)
        selected = solution

        # Pre-compute embeddings (EGNN - no coordinate normalization!)
        self.model_p.pre(pyg_data)

        # Autoregressive partition
        while solution.size(-1) < self.env.raw_problems.size(1):
            if (solution.size(-1) - 1) % self.env.problem_size == 0:
                node_emb, heatmap = self.model_p(solution, visited)
                heatmap = heatmap / (heatmap.min() + 1e-5)
                heatmap = self.model_p.reshape(pyg_data, heatmap) + 1e-5

            row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).squeeze(1)
            row = row * (1 - visited)
            dist = Categorical(row)
            item = dist.sample()
            log_prob = dist.log_prob(item)

            selected = item[:, None]
            logp += log_prob
            visited = visited.scatter(-1, selected, 1)
            solution = torch.cat((solution, selected), dim=-1)

        # Solve sub-problems
        loss_t_total = 0
        for i in range(2):  # Two-phase rolling
            roll = self.env.problem_size // 2
            solution = solution.roll(dims=1, shifts=roll)

            n_tsps_per_route = solution.view(solution.size(0), -1, self.env.problem_size)
            tsp_insts = self.env.raw_problems[:, None, :].repeat(solution.size(0), n_tsps_per_route.size(1), 1, 1)
            tsp_insts = tsp_insts.gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 2))
            tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))

            solution_now = torch.arange(tsp_insts_now.size(-2), device=self.device)[None, :].expand(
                tsp_insts_now.size(0), -1
            )[:, None, :]
            reward_now = self.env.get_open_travel_distance(tsp_insts_now, solution_now)

            new_batch_size = tsp_insts_now.size(0)

            # GEPNet: No coordinate transformation needed!
            self.env.load_problems(new_batch_size, tsp_insts_now)
            reset_state, _, _ = self.env.reset()
            self.model_t.pre_forward(reset_state)

            # POMO rollout
            prob_list = torch.zeros(size=(new_batch_size, self.env.pomo_size, 0), device=self.device)
            state, reward, done = self.env.pre_step()

            while not done:
                cur_dist = self.env.get_local_feature()
                selected_t, prob = self.model_t(state, cur_dist)
                state, reward, done = self.env.step(selected_t)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            # Final step
            selected_t = torch.cat([
                (self.env.problem_size - 1) * torch.ones(self.env.pomo_size // 2, device=self.device)[None, :].expand(new_batch_size, -1),
                torch.zeros(self.env.pomo_size // 2, device=self.device)[None, :].expand(new_batch_size, -1)
            ], dim=-1).long()
            state, reward, done = self.env.step(selected_t)

            # Sub-solver loss
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            loss_t = (-advantage * prob_list.log().sum(dim=2)).mean()

            self.model_t.zero_grad()
            loss_t.backward()
            self.optimizer_t.step()
            loss_t_total += loss_t.item()

            # Update solution
            reward = self.env.get_open_travel_distance(tsp_insts_now, self.env.selected_node_list)
            tag = reward.view(batch_size, self.env.sample_size, -1, self.env.pomo_size).min(-1)[1]
            tag = tag[..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
            tag_solution = self.env.selected_node_list.view(
                batch_size, self.env.sample_size, -1, self.env.pomo_size, self.env.problem_size
            ).gather(-2, tag).squeeze()

            reversed_tag_solution = torch.flip(tag_solution, dims=[2])
            tag_solution[tag.squeeze() >= self.env.pomo_size / 2] = reversed_tag_solution[tag.squeeze() >= self.env.pomo_size / 2]

            r = (reward.min(1)[0] > reward_now.squeeze()).view(self.env.sample_size, -1, 1)
            r = r.expand(-1, -1, tsp_insts_now.size(-2))
            tag_solution[r] = solution_now.view(self.env.sample_size, -1, tsp_insts_now.size(-2))[r]

            merge_solution = n_tsps_per_route.gather(-1, tag_solution).view(solution.size(0), -1)
            solution = merge_solution.clone()

        # Partition loss
        merge_reward = -1 * self.env._get_travel_distance(self.env.raw_problems, solution)
        advantage2 = merge_reward - merge_reward.float().mean(dim=1, keepdims=True)
        loss_partition = (-advantage2 * logp).mean()

        max_pomo_reward, _ = merge_reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model_p.zero_grad()
        loss_partition.backward()
        self.optimizer_p.step()

        return score_mean.item(), loss_t_total / 2, loss_partition.item()

    def _gen_pyg_data(self, coors, k_sparse=100):
        """
        Generate PyG data with E(2)-invariant features.

        Node features: Constant ones (E(2)-invariant)
        Edge features: Negative distances (E(2)-invariant)
        Coordinates: Passed separately for EGNN distance computation

        Note: Raw (x, y) coordinates are NOT used as node features because
        they change under rotation, breaking E(2)-equivariance.
        """
        bs = coors.size(0)
        n_nodes = coors.size(1)

        # Distance matrix (E(2)-invariant!)
        cos_mat = -1 * torch.cdist(coors, coors, p=2)

        topk_values, topk_indices = torch.topk(cos_mat, k=k_sparse, dim=2, largest=True)

        edge_index = torch.cat([
            torch.repeat_interleave(torch.arange(n_nodes, device=coors.device), repeats=k_sparse)[None, :].repeat(bs, 1)[:, None, :],
            topk_indices.view(bs, -1)[:, None, :]
        ], dim=1)

        idx = torch.arange(bs, device=coors.device)[:, None, None].repeat(1, n_nodes, k_sparse)
        edge_attr1 = topk_values.reshape(bs, -1, 1)
        edge_attr2 = cos_mat[idx.view(bs, -1), edge_index[:, 0], edge_index[:, 1]].reshape(bs, k_sparse * n_nodes, 1)
        edge_attr = torch.cat([edge_attr1, edge_attr2], dim=2)

        # Node features: constant ones (E(2)-invariant!)
        # All nodes are indistinguishable by features; EGNN learns from graph structure.
        x = torch.ones(n_nodes, 1, device=coors.device)

        pyg_data = Data(
            x=x,
            edge_index=edge_index[0],
            edge_attr=edge_attr[0],
            pos=coors[0]  # Coordinates for EGNN distance computation (NOT as features)
        )
        return pyg_data

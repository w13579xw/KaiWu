#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 10  # [修改此处] 从 3 改为 10，获取完整的 21x21 视野

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # [新增] 历史轨迹与地图记忆
        self.visited_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.last_pos = (0, 0)
        self.visited_count = 0
        
        # [新增] 动态实体距离追踪
        self.nearest_charger_dist = 200.0
        self.last_charger_dist = 200.0
        self.nearest_npc_dist = 200.0
        
        # [新增] 提取的实体坐标列表
        self.charger_coords = []
        self.npc_coords = []

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # [新增] 记录上一帧的电量，用于计算稀疏奖励
        if not hasattr(self, 'last_battery'):
            self.last_battery = int(hero["battery"])
        else:
            self.last_battery = self.battery

        # Battery / 电量
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

        # [新增] 解析充电桩与NPC位置
        self.charger_coords = []
        if "chargers" in frame_state:
            for charger in frame_state["chargers"]:
                self.charger_coords.append((int(charger["pos"]["x"]), int(charger["pos"]["z"])))
                
        self.npc_coords = []
        if "official_robots" in frame_state: # 或者 "npcs"
            for npc in frame_state["official_robots"]:
                self.npc_coords.append((int(npc["pos"]["x"]), int(npc["pos"]["z"])))

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """
        升级版局部视野特征（3通道 x 21 x 21）：
        Channel 0: 原始地形与污渍
        Channel 1: 充电桩相对位置投影
        Channel 2: 官方机器人(NPC)相对位置投影
        """
        center = self.VIEW_HALF  # 10
        h = self.LOCAL_HALF      # 10 (保证是 21x21 的完整视野)
        
        # 截取基础地形图 (21x21)
        base_crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        channel_0 = base_crop / 2.0  # 归一化
        
        # 初始化通道 1 和 2 (尺寸与视野一致)
        channel_1_chargers = np.zeros_like(channel_0)
        channel_2_npcs = np.zeros_like(channel_0)
        
        hx, hz = self.cur_pos
        
        # 将全局充电桩坐标投影到局部 21x21 视野中
        if hasattr(self, 'charger_coords'):
            for cx, cz in self.charger_coords:
                # 计算相对坐标
                rel_x = cx - hx
                rel_z = cz - hz
                # 如果在 21x21 的视野范围内，则在通道1上打上高光
                if -h <= rel_x <= h and -h <= rel_z <= h:
                    # 矩阵的行列对应：行对应x方向，列对应z方向（需与基础地形坐标系一致）
                    channel_1_chargers[center + rel_x, center + rel_z] = 1.0
                    
        # 将全局NPC坐标投影到局部 21x21 视野中
        if hasattr(self, 'npc_coords'):
            for nx, nz in self.npc_coords:
                rel_x = nx - hx
                rel_z = nz - hz
                if -h <= rel_x <= h and -h <= rel_z <= h:
                    channel_2_npcs[center + rel_x, center + rel_z] = 1.0

        # 拼接为 3 通道特征，并展平返回
        # 最终维度: 3 * 21 * 21 = 1323 维
        multi_channel_view = np.stack([channel_0, channel_1_chargers, channel_2_npcs], axis=0)
        return multi_channel_view.flatten()

    def _get_global_state_feature(self):
        """Global state feature (12D).

        全局状态特征（12D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
          [7]  ray_E_dirt        east ray distance / 向右（x+）方向
          [8]  ray_S_dirt        south ray distance / 向下（z+）方向
          [9]  ray_W_dirt        west ray distance / 向左（x-）方向
          [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(x - (hx - self.VIEW_HALF), 0, 20), np.clip(z - (hz - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        # 1. 充电桩特征 (最近充电桩距离)
        self.last_charger_dist = self.nearest_charger_dist
        if self.charger_coords:
            dists = [np.sqrt((cx - hx)**2 + (cz - hz)**2) for cx, cz in self.charger_coords]
            self.nearest_charger_dist = min(dists)
        charger_dist_norm = _norm(self.nearest_charger_dist, self.GRID_SIZE)
        
        # 2. NPC 特征 (最近NPC距离与接近警报)
        npc_alert = 0.0
        if self.npc_coords:
            dists = [np.sqrt((nx - hx)**2 + (nz - hz)**2) for nx, nz in self.npc_coords]
            self.nearest_npc_dist = min(dists)
            if self.nearest_npc_dist <= 5.0:  # 距离小于5格拉响警报
                npc_alert = 1.0
        npc_dist_norm = _norm(self.nearest_npc_dist, self.GRID_SIZE)
        
        # 3. 轨迹与探索特征
        is_new_area = 1.0 if self.visited_map[hx, hz] == 0 else 0.0
        if is_new_area == 1.0:
            self.visited_map[hx, hz] = 1
            self.visited_count += 1
        visited_ratio = _norm(self.visited_count, self.GRID_SIZE * self.GRID_SIZE)

        # 低电量标志位
        is_low_battery = 1.0 if self.battery < (self.battery_max * 0.25) else 0.0

        # 返回扩充后的 18D 向量
        return np.array(
            [
                step_norm, battery_ratio, cleaning_progress, remaining_dirt,
                pos_x_norm, pos_z_norm, 
                ray_dirt[0], ray_dirt[1], ray_dirt[2], ray_dirt[3], 
                nearest_dirt_norm, dirt_delta,
                # --- 以下为新增的 6D 特征 ---
                charger_dist_norm,   # 最近充电桩归一化距离
                npc_dist_norm,       # 最近官方机器人归一化距离
                npc_alert,           # 危险接近标志位 (0或1)
                is_low_battery,      # 低电量紧急标志位 (0或1)
                visited_ratio,       # 全局探索率
                is_new_area          # 当前格子是否是首次到达
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)

    def feature_process(self, env_obs, last_action):
        """Generate 69D feature vector, legal action mask, and scalar reward.

        生成 69D 特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()  # 12D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr])  # 69D

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        hx, hz = self.cur_pos
        
        # 1. 基础清扫奖励 (提高权重，鼓励核心任务)
        cleaned_this_step = max(0, self.dirt_cleaned - getattr(self, 'last_dirt_cleaned', 0))
        cleaning_reward = 0.1 * cleaned_this_step
        
        # ==========================================
        # 2. [大幅优化] 探索、空走与“已清洁地面”惩罚
        # ==========================================
        explore_reward = 0.0
        idle_penalty = 0.0
        clean_ground_penalty = 0.0
        
        # (1) 探索新区域的强奖励
        if hasattr(self, 'visited_map'):
            if self.visited_map[hx, hz] == 0:
                explore_reward = 0.5  # [加大] 从 0.005 加大到 0.02，强烈鼓励探索
                self.visited_map[hx, hz] = 1 

        # (2) 移动状态的严厉惩罚
        if hasattr(self, 'last_pos'):
            if self.cur_pos == self.last_pos:
                # [加大惩罚] 原地发呆或撞墙的惩罚加倍
                idle_penalty = -0.05  # 从 -0.015 加大到 -0.05
            else:
                # 如果发生了移动，但没扫到垃圾
                if cleaned_this_step == 0:
                    # [加大惩罚] 走在已清洁的地板上，施加持续的“厌恶感”
                    clean_ground_penalty = -0.05  # 从 -0.01 加大到 -0.02
                    
        self.last_pos = self.cur_pos

        # ==========================================
        # 3. 防刷分的充电奖励 (保持之前的修复)
        # ==========================================
        charge_reward = 0.0 
        last_bat = getattr(self, 'last_battery', self.battery)
        
        # 只有低于 25% 才有引导
        if self.battery < (self.battery_max * 0.25):
            if hasattr(self, 'last_charger_dist') and self.nearest_charger_dist < self.last_charger_dist:
                charge_reward += 0.1  
            elif hasattr(self, 'last_charger_dist') and self.nearest_charger_dist > self.last_charger_dist:
                charge_reward -= 0.1  

        # 稀疏奖励：成功充电
        if last_bat > 0 and self.battery > last_bat:
            if last_bat < (self.battery_max * 0.25):
                charge_reward += 30.0  
            else:
                charge_reward -= 0.10

        # ==========================================
        # 4. [加重] NPC 碰撞稀疏惩罚 + 躲避稠密惩罚
        # ==========================================
        safety_penalty = 0.0
        if self.nearest_npc_dist <= 1.0:
            # [极其严厉] 撞上官方机器人，直接视为“死罪”级别的惩罚
            safety_penalty = -500.0  # 从 -20 加大到 -40
        elif self.nearest_npc_dist <= 3.0:
            # [加大缓冲区惩罚] 让它离NPC远一点
            safety_penalty = -0.3    # 从 -0.05 加大到 -0.15
        elif self.nearest_npc_dist <= 6.0:
            # 缓冲区衰减惩罚
            safety_penalty = -0.3 * (6.0 - self.nearest_npc_dist) # 从 -0.01 衰减率加大到 -0.05

        # 5. 存活步数惩罚 (时间成本)
        step_penalty = -0.0005

        # ==========================================
        # 6. [新增] "过劳死"极度惩罚 (Death Penalty)
        # ==========================================
        death_penalty = 0.0
        # 当电量降至0或极低警戒线（假设为0）时，触发致死惩罚
        if self.battery <= 0:
            # 这个数值需要足够大，能够抵消它之前贪婪扫地获得的分数
            # 假设一局最多扫100格，每格0.15分，总收益15分。
            # 这里给 -40，足以让它在价值评估时产生深深的恐惧。
            death_penalty = -500.0  

        # 多头奖励加总
        total_reward = (
            cleaning_reward + 
          #  explore_reward + 
          #  idle_penalty + 
          #  clean_ground_penalty +
          #  charge_reward + 
          #  safety_penalty + 
            step_penalty 
          #  + death_penalty # 加入致死惩罚
        )
        
        return total_reward

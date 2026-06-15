
from typing import Dict, Generator, Iterable, List, Optional

from tspipe.batch_ops import BatchQueue


class PipelineSchedule():
    # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)
    def __init__(self, legacy_tuple=None, terminate=False):
        if legacy_tuple is not None:
            i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop, optimize \
                = legacy_tuple

            self.i = i
            self.j = j
            self.prev_partition = prev_partition
            self.is_target = is_target
            self.batch_idx = batch_idx
            self.view_idx = view_idx
            self.dependencies = dependencies
            self.model_update_ver = model_update_ver
            self.backprop = backprop
            self.optimize = optimize
        self.terminate = terminate

    def get_tuple(self):
        return (
            self.i,
            self.j,
            self.prev_partition,
            self.is_target,
            self.batch_idx,
            self.view_idx,
            self.dependencies,
            self.model_update_ver,
            self.backprop,
            self.optimize)

    @property
    def ubatch_idx(self):
        return self.i

    @property
    def partition_idx(self):
        return self.j

    def __repr__(self) -> str:
        return f"PipelineSchedule<{self.__dict__}>"


def schedule_without_pipeline(devices, batch_index, optimize_len, gpipe_num_ubatch=None) \
        -> Iterable[Iterable[Optional[PipelineSchedule]]]:
    # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)

    num_devices = devices
    # num_batches = int((3 * (num_devices - 1) - optimize_len) / 2)
    if gpipe_num_ubatch is not None:
        num_batches = gpipe_num_ubatch
    else:
        num_batches = num_devices - 1

    schedule = []
    micro_idx = [0 for i in range(num_devices)]
    is_target = [0 for i in range(num_devices)]
    view_idx = [0 for i in range(num_devices)]
    forward = [True for i in range(num_devices)]
    # Forward
    for bat_id in range(4 * num_batches + (num_devices - 1)):
        sched = []
        for dev_id in range(num_devices):
            if micro_idx[dev_id] >= num_batches:
                micro_idx[dev_id] -= num_batches
            prev = dev_id - 1 if dev_id > 0 else None
            if is_target[dev_id] >= num_batches * 4:
                is_target[dev_id] -= 4 * num_batches
                forward[dev_id] = False
            is_target_bool = is_target[dev_id] < 2 * num_batches
            if view_idx[dev_id] >= num_batches * 2:
                view_idx[dev_id] -= 2 * num_batches
            view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0
            if micro_idx[dev_id] == 0 and view_idx_int == 0:
                model_update_ver = batch_index - 1
            else:
                model_update_ver = None
            if dev_id <= bat_id and forward[dev_id]:
                #            (i,         j,     prev_partition, is_target,      batch_idx,   view_idx, dependencies, model_update_ver, backprop, optimize) # noqa :E203
                block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index, view_idx_int, None, model_update_ver, False, False]) # noqa :E203
                micro_idx[dev_id] = micro_idx[dev_id] + 1
                view_idx[dev_id] = view_idx[dev_id] + 1
                is_target[dev_id] = is_target[dev_id] + 1
            else:
                block = None
            sched.append(PipelineSchedule(block) if block is not None else None)
        schedule.append(sched)
    grad_micro_idx = [num_batches - 1 for i in range(num_devices)]
    grad_view_idx = [num_batches * 2 - 1 for i in range(num_devices)]
    backward_start = num_devices - 1

    # Backward
    for bat_id in range(2 * num_batches + (num_devices - 1) + 1):
        sched = []
        for dev_id in range(num_devices):
            if grad_micro_idx[dev_id] < 0:
                grad_micro_idx[dev_id] += num_batches
            prev = dev_id + 1 if dev_id < num_devices - 1 else None
            grad_view_idx_int = 1 if grad_view_idx[dev_id] >= num_batches else 0
            if dev_id >= backward_start and grad_view_idx[dev_id] >= 0:
                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               grad_view_idx_int, None, None, True, False])
                grad_micro_idx[dev_id] = grad_micro_idx[dev_id] - 1
                grad_view_idx[dev_id] = grad_view_idx[dev_id] - 1
            else:
                if grad_view_idx[dev_id] == -1:
                    block = tuple([0, dev_id, None, False, batch_index, 0, None, None, False, True])
                    grad_view_idx[dev_id] -= 1
                else:
                    block = None
            sched.append(PipelineSchedule(block) if block is not None else None)
        schedule.append(sched)
        backward_start -= 1
    return schedule


def schedule_start(devices, batch_index, micro_idx, is_target, view_idx, optimize_len, skip=False) \
        -> Iterable[Iterable[Optional[PipelineSchedule]]]:
    num_devices = devices
    num_batches = num_devices - 1
    start = []
    for bat_id in range(2 * num_batches):
        schedule = []
        for dev_id in range(num_devices):

            if micro_idx[dev_id] >= num_batches:
                micro_idx[dev_id] -= num_batches

            prev = dev_id - 1 if dev_id > 0 else None

            if is_target[dev_id] >= num_batches * 4:
                is_target[dev_id] -= num_batches * 4

            is_target_bool = is_target[dev_id] < 2 * num_batches

            if view_idx[dev_id] >= num_batches * 2:
                view_idx[dev_id] -= num_batches * 2

            view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0
            if not skip:
                if micro_idx[dev_id] == 0 and view_idx_int == 0 and is_target_bool:
                    model_update_ver = batch_index - 1
                else:
                    model_update_ver = None
            else:
                model_update_ver = None

            if dev_id <= bat_id:
                block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index, view_idx_int, None,
                               model_update_ver, False, False])
                micro_idx[dev_id] = micro_idx[dev_id] + 1
                view_idx[dev_id] = view_idx[dev_id] + 1
                is_target[dev_id] = is_target[dev_id] + 1
            else:
                block = None
            schedule.append(PipelineSchedule(block) if block is not None else None)
        start.append(schedule)
    return start


def schedule_repeat(devices, batch_index, micro_idx, is_target, view_idx, skip_target=False, skip_online=False) \
        -> Iterable[Iterable[Optional[PipelineSchedule]]]:
    # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)
    num_devices = devices
    num_batches = num_devices - 1
    repeat = []

    backprop = [-1 for i in range(num_devices)]
    propagate = [False for i in range(num_devices)]
    grad_view_idx = [0 for i in range(num_devices)]
    grad_micro_idx = [num_batches - 1 for i in range(num_devices)]
    batch_count = []
    for i in range(num_devices):
        batch_count.append(-i)

    for i in range(num_batches * 6):
        schedule = []
        dependency = None
        for dev_id in range(num_devices):
            if backprop[dev_id] == -1:

                if micro_idx[dev_id] >= num_batches:
                    micro_idx[dev_id] -= num_batches

                # previous Partitions
                prev = dev_id - 1 if dev_id > 0 else None

                if is_target[dev_id] >= num_batches * 4:
                    is_target[dev_id] -= num_batches * 4

                is_target_bool = is_target[dev_id] < 2 * num_batches

                if view_idx[dev_id] >= num_batches * 2:
                    view_idx[dev_id] -= num_batches * 2

                view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0

                # Dependency
                if dev_id == num_devices - 1 and is_target[dev_id] == 4*num_batches - 1:
                    dependency = batch_index
                    backprop[dev_id] = 2*num_batches - 1

                # Model update Version
                if skip_target:
                    if micro_idx[dev_id] == 0 and not is_target_bool and batch_count[dev_id] == 0:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None
                elif skip_online:
                    if micro_idx[dev_id] == 0 and is_target_bool and batch_count[dev_id] == 2*num_batches:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None
                else:
                    if micro_idx[dev_id] == 0 and batch_count[dev_id] == 0:
                        model_update_ver = batch_index - 2
                    elif micro_idx[dev_id] == 0 and batch_count[dev_id] == 2*num_batches:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None

                if batch_count[dev_id] >= 2*num_batches:
                    block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index + 1,
                                   view_idx_int, dependency, model_update_ver, False])
                else:
                    block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index,
                                   view_idx_int, dependency, model_update_ver, False])

                micro_idx[dev_id] = micro_idx[dev_id] + 1
                view_idx[dev_id] = view_idx[dev_id] + 1
                is_target[dev_id] = is_target[dev_id] + 1
                batch_count[dev_id] = batch_count[dev_id] + 1
            else:

                if grad_micro_idx[dev_id] < 0:
                    grad_micro_idx[dev_id] += num_batches

                prev = dev_id + 1 if dev_id < num_devices - 1 else None

                if dev_id > 0 and not propagate[dev_id]:
                    backprop[dev_id - 1] = 2*num_batches - 1
                    propagate[dev_id] = True

                grad_view_idx_int = 1 if grad_view_idx[dev_id] < num_batches else 0

                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               grad_view_idx_int, None, None, True])

                backprop[dev_id] = backprop[dev_id] - 1
                grad_micro_idx[dev_id] = grad_micro_idx[dev_id] - 1
                grad_view_idx[dev_id] = grad_view_idx[dev_id] + 1

            schedule.append(PipelineSchedule(block) if block is not None else None)
        repeat.append(schedule)
    return repeat


def new_schedule_start(devices, batch_index, micro_idx, is_target, view_idx, optimize, batch_count, optimize_len):
    num_devices = devices
    num_batches = num_devices - 1
    repeat = []

    backprop = [-1 for i in range(num_devices)]
    propagate = [0 for i in range(num_devices)]
    grad_view_idx = [0 for i in range(num_devices)]
    grad_micro_idx = [2 * num_batches - 1 for i in range(num_devices)]
    for bat_id in range(8 * num_batches + optimize_len):
        schedule = []
        dependency = None
        for dev_id in range(num_devices):
            if backprop[dev_id] == -1: # Forward 단계 스케줄링 (해당 파티션이 아직 backward를 시작하지 않은 경우)

                if optimize[dev_id] > 0:

                    if optimize[dev_id] == optimize_len:
                        block = tuple([0, dev_id, 0, False, batch_index, 0, None, None, False, True])
                    optimize[dev_id] -= 1
                else:
                    if micro_idx[dev_id] >= num_batches:
                        micro_idx[dev_id] -= num_batches

                    # previous Partitions
                    prev = dev_id - 1 if dev_id > 0 else None

                    if is_target[dev_id] >= num_batches * 4:
                        is_target[dev_id] -= num_batches * 4

                    is_target_bool = is_target[dev_id] < 2 * num_batches

                    if view_idx[dev_id] >= num_batches * 2:
                        view_idx[dev_id] -= num_batches * 2

                    view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0

                    # Dependency. 마지막 파티션이 마지막 forward를 처리할 경우 dependency 설정 
                    if dev_id == num_devices - 1 and is_target[dev_id] == 4*num_batches - 1:
                        dependency = batch_index
                        backprop[dev_id] = 4*num_batches - 1 # (나중 backward 시작 시점)

                    # Model Update Version
                    if micro_idx[dev_id] == 0 and batch_count[dev_id] % (2*num_batches) == 0:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None

                    if batch_count[dev_id] >= 2*num_batches:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index + 1,
                                       view_idx_int, dependency, model_update_ver, False, False])
                    else:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index,
                                       view_idx_int, dependency, model_update_ver, False, False])

                    micro_idx[dev_id] = micro_idx[dev_id] + 1
                    view_idx[dev_id] = view_idx[dev_id] + 1
                    is_target[dev_id] = is_target[dev_id] + 1
                    batch_count[dev_id] = batch_count[dev_id] + 1

                    if batch_count[dev_id] >= 4*num_batches and optimize[dev_id] == 0:
                        optimize[dev_id] = optimize_len
                        batch_count[dev_id] = 0
            else: # Backward 단계 스케줄링 (해당 파티션이 backward 단계에 들어간 경우)
                if grad_micro_idx[dev_id] < 0:
                    grad_micro_idx[dev_id] += 2 * num_batches

                prev = dev_id + 1 if dev_id < num_devices - 1 else None

                if grad_view_idx[dev_id] >= num_batches * 4:
                    grad_view_idx[dev_id] = 0

                if dev_id > 0 and propagate[dev_id] < 1:
                    propagate[dev_id] += 1
                    if propagate[dev_id] == 1:
                        backprop[dev_id - 1] = 4*num_batches - 1

                grad_view_idx_int = 1 if grad_view_idx[dev_id] < 2 * num_batches else 0

                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               grad_view_idx_int, None, None, True, False])

                backprop[dev_id] = backprop[dev_id] - 1
                grad_micro_idx[dev_id] = grad_micro_idx[dev_id] - 1
                grad_view_idx[dev_id] = grad_view_idx[dev_id] + 1

            schedule.append(PipelineSchedule(block) if block is not None else None)
        repeat.append(schedule)
    return repeat # -> List[List[PipelineSchedule]]바깥 리스트: 각 clock (시간 스텝)


def new_schedule_repeat(devices, batch_index, micro_idx, is_target, view_idx, optimize, batch_count, optimize_len):
    num_devices = devices
    num_batches = num_devices - 1
    repeat = []

    backprop = [-1 for i in range(num_devices)]
    propagate = [0 for i in range(num_devices)]
    grad_view_idx = [0 for i in range(num_devices)]
    grad_micro_idx = [2 * num_batches - 1 for i in range(num_devices)]
    reset = False

    for bat_id in range(8 * num_batches + optimize_len):
        schedule = []
        dependency = None
        for dev_id in range(num_devices):
            if backprop[dev_id] == -1:
                if optimize[dev_id] > 0:
                    if optimize[dev_id] == optimize_len:
                        if reset:
                            block = tuple([0, dev_id, 0, False, batch_index, 0, None, None, False, True])
                        else:
                            block = tuple([0, dev_id, 0, False, batch_index - 1, 0, None, None, False, True])
                    else:
                        block = None
                    optimize[dev_id] -= 1
                else:
                    if micro_idx[dev_id] >= num_batches:
                        micro_idx[dev_id] -= num_batches
                        if dev_id == num_devices - 1:
                            reset = True

                    # previous Partitions
                    prev = dev_id - 1 if dev_id > 0 else None

                    if is_target[dev_id] >= num_batches * 4:
                        is_target[dev_id] -= num_batches * 4

                    is_target_bool = is_target[dev_id] < 2 * num_batches

                    if view_idx[dev_id] >= num_batches * 2:
                        view_idx[dev_id] -= num_batches * 2

                    view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0

                    # Dependency
                    if dev_id == num_devices - 1 and is_target[dev_id] == 4*num_batches - 1:
                        dependency = batch_index
                        backprop[dev_id] = 4*num_batches - 1

                    # Model Update Ver
                    if micro_idx[dev_id] == 0 and batch_count[dev_id] % (2*num_batches) == 0:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None

                    if batch_count[dev_id] >= 2*num_batches and reset:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index + 1,
                                       view_idx_int, dependency, model_update_ver, False, False])
                    else:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index,
                                       view_idx_int, dependency, model_update_ver, False, False])

                    micro_idx[dev_id] = micro_idx[dev_id] + 1
                    view_idx[dev_id] = view_idx[dev_id] + 1
                    is_target[dev_id] = is_target[dev_id] + 1
                    batch_count[dev_id] = batch_count[dev_id] + 1

                    if batch_count[dev_id] >= 4*num_batches and optimize[dev_id] == 0:
                        optimize[dev_id] = optimize_len
                        batch_count[dev_id] = 0
            else:
                if grad_micro_idx[dev_id] < 0:
                    grad_micro_idx[dev_id] += 2 * num_batches

                prev = dev_id + 1 if dev_id < num_devices - 1 else None

                if grad_view_idx[dev_id] >= num_batches * 4:
                    grad_view_idx[dev_id] = 0

                if dev_id > 0 and propagate[dev_id] < 1:
                    propagate[dev_id] += 1
                    if propagate[dev_id] == 1:
                        backprop[dev_id - 1] = 4*num_batches - 1

                grad_view_idx_int = 1 if grad_view_idx[dev_id] < 2 * num_batches else 0

                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               grad_view_idx_int, None, None, True, False])

                backprop[dev_id] = backprop[dev_id] - 1
                grad_micro_idx[dev_id] = grad_micro_idx[dev_id] - 1
                grad_view_idx[dev_id] = grad_view_idx[dev_id] + 1

            schedule.append(PipelineSchedule(block) if block is not None else None)
        repeat.append(schedule)
    return repeat


def schedule_generator_mp(devices, batch_idx):
    n = devices
    m = 1
    batch_index = batch_idx
    # (i, j, prev_partition, is_target, batch_index,
    #  view_idx, dependencies, model_update_ver, backprop)
    repeat = []
    is_target = [1 for i in range(n)]
    view_idx = [0 for i in range(n)]
    batch_count = [0 for i in range(n)]
    grad_count = [0 for i in range(n)]
    propagate = [False for i in range(n)]
    back_view_idx = [1 for i in range(n)]
    forward = True
    for k in range(2*n + 4):
        schedule = []
        for j in range(n):
            if forward:
                prev = j - 1 if j > 0 else None

                if is_target[j] < - 2 * m:
                    is_target[j] = 1

                if view_idx[j] > 1:
                    view_idx[j] = 0

                if batch_count[j] == 4*m - 1 and j == n - 1:
                    forward = False
                    propagate[j] = True

                if batch_count[j] == 0:
                    model_update_ver = batch_index - 2
                elif batch_count[j] == 2 * m:
                    model_update_ver = batch_index - 2
                else:
                    model_update_ver = None

                if j <= k and batch_count[j] <= 4*m - 1:
                    block = tuple([0, j, prev, is_target[j] >= 0, batch_index, view_idx[j],
                                   None, model_update_ver, False])
                    is_target[j] -= 1
                    view_idx[j] += 1
                    batch_count[j] += 1
                else:
                    block = None
            else:
                prev = j + 1 if j < n - 1 else None

                if back_view_idx[j] > 1:
                    back_view_idx[j] = 0

                if propagate[j] and grad_count[j] < 2*m:
                    block = tuple([0, j, prev, False, batch_index, back_view_idx[j], None, None, True])
                    grad_count[j] += 1
                    if j != 0:
                        propagate[j-1] = True
                    back_view_idx[j] += 1
                else:
                    block = None
            schedule.append(block)
        repeat.append(PipelineSchedule(*schedule))
    return repeat


def sanity_check(sched_gen_func):
    def wrap(s):
        schedule_history = set()
        schedule_gen = sched_gen_func(s)
        for schedule in schedule_gen:
            for sched in schedule:
                if sched is None:
                    continue
                if sched in schedule_history:
                    raise ValueError(f"This task has been already scheduled : {sched}")
                else:
                    schedule_history.add(sched)
            yield schedule
    return wrap


class TSPipeScheduler():
    def __init__(self, config: Dict, input_batch_queue: BatchQueue, num_partitions: int):
        self.input_batch_queue = input_batch_queue
        self.num_partitions = num_partitions

        self.gpipe_emulation_enabled = config['gpipe_emulation']['enabled']
        if self.gpipe_emulation_enabled:
            self.num_skip_staleness = None
            self.gpipe_num_ubatch = config['gpipe_emulation']['num_ubatch']
        else:
            self.num_skip_staleness = config['optimizer']['num_skip_initial_staleness']
            self.gpipe_num_ubatch = None
            
        # Failover 관련 속성 추가
        self.original_num_partitions = num_partitions
        self.current_partition_config = None
        self.failover_active = False
        self.reconfiguration_pending = False

    def update_partition_config(self, new_num_partitions: int, new_config: Dict):
        """파티션 설정 업데이트 (Failover 시 호출)"""
        from tspipe.logger import Log
        
        Log.i(f"🔀 Updating partition configuration: {self.num_partitions} -> {new_num_partitions}")
        
        old_partitions = self.num_partitions
        self.num_partitions = new_num_partitions
        self.current_partition_config = new_config
        self.failover_active = True
        
        Log.i(f"✅ Partition configuration updated: partitions={new_num_partitions}")
        
        return {
            'old_partitions': old_partitions,
            'new_partitions': new_num_partitions,
            'new_config': new_config
        }

    def dynamic_repartition(self, available_gpu_count: int, model_config: Dict) -> Dict:
        """동적 모델 재분할 (GPU 실패 시)"""
        from tspipe.logger import Log
        
        Log.i(f"🧮 Computing dynamic repartition for {available_gpu_count} GPUs")
        
        if available_gpu_count <= 0:
            raise ValueError("Cannot repartition with 0 GPUs")
        
        # 기본 균등 분할 전략
        new_config = self._compute_equal_partition(available_gpu_count, model_config)
        
        # 더 정교한 분할 알고리즘 적용 (옵션)
        optimized_config = self._optimize_partition_with_dp(available_gpu_count, model_config)
        if optimized_config:
            new_config = optimized_config
        
        Log.i(f"✅ Dynamic repartition computed: {new_config}")
        return new_config

    def _compute_equal_partition(self, num_gpus: int, model_config: Dict) -> Dict:
        """균등 분할 계산"""
        online_layers = model_config.get('online', [])
        target_layers = model_config.get('target', [])
        
        total_online = sum(online_layers) if online_layers else 0
        total_target = sum(target_layers) if target_layers else 0
        
        # 균등 분할
        online_per_gpu = max(1, total_online // num_gpus)
        target_per_gpu = max(1, total_target // num_gpus)
        
        new_online = [online_per_gpu] * num_gpus
        new_target = [target_per_gpu] * num_gpus
        
        # 나머지 레이어들을 마지막 GPU에 할당
        remaining_online = total_online - sum(new_online)
        remaining_target = total_target - sum(new_target)
        
        if remaining_online > 0:
            new_online[-1] += remaining_online
        if remaining_target > 0:
            new_target[-1] += remaining_target
        
        return {
            'online': new_online,
            'target': new_target
        }

    def _optimize_partition_with_dp(self, num_gpus: int, model_config: Dict) -> Optional[Dict]:
        """동적 프로그래밍을 이용한 최적 분할 (간단한 버전)"""
        try:
            # 실제 DP 알고리즘 구현은 복잡하므로, 여기서는 간단한 휴리스틱 적용
            # 논문의 알고리즘을 간소화한 버전
            
            online_layers = model_config.get('online', [])
            target_layers = model_config.get('target', [])
            
            if not online_layers or not target_layers:
                return None
            
            total_online = sum(online_layers)
            total_target = sum(target_layers)
            
            # GPU 수가 적을 때는 비율을 유지하면서 분할
            if num_gpus < len(online_layers):
                # 원본 비율 기반 분할
                online_ratios = [x / total_online for x in online_layers]
                target_ratios = [x / total_target for x in target_layers]
                
                # 새로운 파티션 수에 맞게 그룹핑
                groups_per_gpu = len(online_layers) // num_gpus
                remainder = len(online_layers) % num_gpus
                
                new_online = []
                new_target = []
                
                start_idx = 0
                for gpu_idx in range(num_gpus):
                    group_size = groups_per_gpu + (1 if gpu_idx < remainder else 0)
                    end_idx = start_idx + group_size
                    
                    # 해당 그룹의 레이어 합산
                    online_sum = sum(online_layers[start_idx:end_idx])
                    target_sum = sum(target_layers[start_idx:end_idx])
                    
                    new_online.append(online_sum)
                    new_target.append(target_sum)
                    
                    start_idx = end_idx
                
                return {
                    'online': new_online,
                    'target': new_target
                }
            
        except Exception as e:
            from tspipe.logger import Log
            Log.e(f"DP optimization failed: {e}")
            
        return None

    def handle_partition_failure(self, failed_partition_ids: List[int]) -> Dict:
        """파티션 실패 처리"""
        from tspipe.logger import Log
        
        Log.w(f"🚨 Handling partition failures: {failed_partition_ids}")
        
        remaining_partitions = self.num_partitions - len(failed_partition_ids)
        
        if remaining_partitions <= 0:
            raise RuntimeError("All partitions have failed!")
        
        # 스케줄 재조정 필요 플래그 설정
        self.reconfiguration_pending = True
        
        Log.i(f"📋 Partition failure handling complete. Remaining: {remaining_partitions}")
        
        return {
            'failed_partitions': failed_partition_ids,
            'remaining_partitions': remaining_partitions,
            'reconfiguration_needed': True
        }

    def wait_for_current_batch_completion(self):
        """현재 배치 완료 대기 (Failover 시 안전한 재구성을 위해)"""
        from tspipe.logger import Log
        import time
        
        Log.i("⏳ Waiting for current batch completion before reconfiguration...")
        
        # TODO: 실제 배치 완료 시그널 구현
        # 현재는 임시로 고정 대기 시간 사용
        time.sleep(2.0)
        
        Log.i("✅ Current batch completion wait finished")

    def reset_failover_state(self):
        """Failover 상태 리셋"""
        from tspipe.logger import Log
        
        self.failover_active = False
        self.reconfiguration_pending = False
        self.current_partition_config = None
        
        Log.i("🔄 Scheduler failover state reset")

    def get_failover_status(self) -> Dict:
        """Failover 상태 반환"""
        return {
            'failover_active': self.failover_active,
            'reconfiguration_pending': self.reconfiguration_pending,
            'original_partitions': self.original_num_partitions,
            'current_partitions': self.num_partitions,
            'current_config': self.current_partition_config
        }

    # 스케줄 내 작업 유효성 체크: 스케줄에서 요구하는 batch가 input queue에 들어올 때까지 blocking
    def throttle_until_batch_available(self, lst_sched: Iterable[Optional[PipelineSchedule]]):
        # throttle scheduling
        for subsched in lst_sched:
            if subsched is not None:
                self.input_batch_queue.wait_batch(subsched.batch_idx)

    @sanity_check
    def schedule_generator(self) -> Generator[List[Optional[PipelineSchedule]], None, None]:
        """Generates schedules for each clock cycle."""
        # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)
        n = num_devices = self.num_partitions
        optimize_len = 1

        schedule = None
        if self.gpipe_emulation_enabled:
            batch_idx = 0
            while True:
                batch_idx += 1
                schedule = schedule_without_pipeline(n, batch_idx, optimize_len, gpipe_num_ubatch=self.gpipe_num_ubatch)
                for sched in schedule:
                    self.throttle_until_batch_available(sched)
                    yield sched

        # 초기 warm-up 단계 (Staleness 허용)
        for batch_idx in range(1, self.num_skip_staleness + 1):
            schedule = schedule_without_pipeline(n, batch_idx, optimize_len)
            for sched in schedule:
                self.throttle_until_batch_available(sched)
                yield sched
        batch_index = self.num_skip_staleness + 1

        # Staleness 허용이 끝난 후, 정상적인 pipeline scheduling 시작
        # micro_idx, is_target, view_idx, optimize 상태 변수를 기반으로 초기 스케줄 구성
        micro_idx, view_idx, is_target, optimize = [0] * n, [0] * n, [0] * n, [0] * n
        batch_count = [] # 각 파티션별로 forward/backward 작업이 몇 번 수행되었는지 카운트
        for i in range(num_devices):
            batch_count.append(-i)

        # forward-only warm-up (적재)
        schedule = schedule_start(n, batch_index, micro_idx, is_target, view_idx, optimize_len)
        for sched in schedule:
            self.throttle_until_batch_available(sched)
            yield sched
        
        # forward + backward 초기화 (시동)
        schedule = new_schedule_start(n, batch_index, micro_idx, is_target, view_idx,
                                      optimize, batch_count, optimize_len)
        for sched in schedule:
            self.throttle_until_batch_available(sched)
            yield sched
        batch_index += 1

        # 본격적으로 pipeline 학습 반복 시작. 매 cycle에 대해 파티션별로 forward/backward 작업 할당
        while True:
            schedule = new_schedule_repeat(n, batch_index, micro_idx, is_target, view_idx,
                                           optimize, batch_count, optimize_len)
            batch_index += 1
            for sched in schedule:
                self.throttle_until_batch_available(sched)
                yield sched


if __name__ == "__main__":
    n = 4
    schedule = 0
    for batch_idx in range(1, 3):
        print()
        schedule = schedule_without_pipeline(n, batch_idx)
    print()
    micro_idx = [0 for i in range(n)]
    view_idx = [0 for i in range(n)]
    is_target = [0 for i in range(n)]
    optimize = [0 for i in range(n)]

    schedule = schedule_start(n, 3, micro_idx, is_target, view_idx)
    print()
    schedule = new_schedule_start(n, 3, micro_idx, is_target, view_idx, optimize)
    print()
    schedule = new_schedule_repeat(n, 4, micro_idx, is_target, view_idx, optimize)
    print()
    schedule = new_schedule_repeat(n, 5, micro_idx, is_target, view_idx, optimize)

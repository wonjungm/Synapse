
from typing import Dict, Generator, Iterable, List, Optional

from tspipe.batch_ops import BatchQueue


class PipelineScheduleKD():
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
        return f"PipelineScheduleKD<{self.__dict__}>"

def schedule_gpipe(devices, batch_index, optimize_len, gpipe_num_ubatch=None) \
        -> Iterable[Iterable[Optional[PipelineScheduleKD]]]:
    # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)

    num_devices = devices
    num_ubatch = gpipe_num_ubatch if gpipe_num_ubatch is not None else num_devices - 1

    schedule = []
    micro_idx = [0 for _ in range(num_devices)]		# current microbatch index (0 to num_ubatch-1)
    is_target = [0 for _ in range(num_devices)]		# Count of microbatches processed to determine if view is teacher/student. from 0 to 2*num_ubatch each value
    forward = [True for _ in range(num_devices)]	# forward active flag
    
    # Forward
    for bat_id in range(2 * num_ubatch + (num_devices - 1)):
        sched = []
        for dev_id in range(num_devices):
            if micro_idx[dev_id] >= num_ubatch:
                micro_idx[dev_id] -= num_ubatch # keeps micro index in range (0 to num_ubatch-1)
                
            prev = dev_id - 1 if dev_id > 0 else None
            
            if is_target[dev_id] >= num_ubatch * 2:
                is_target[dev_id] -= 2 * num_ubatch
                forward[dev_id] = False
            is_target_bool = is_target[dev_id] < num_ubatch
            
            if micro_idx[dev_id] == 0 and is_target[dev_id] == 0:
                model_update_ver = batch_index - 1
            else:
                model_update_ver = None
            if dev_id <= bat_id and forward[dev_id]:
                #            (i,         j,     prev_partition, is_target,      batch_idx,   view_idx, dependencies, model_update_ver, backprop, optimize) # noqa :E203
                block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index, 0, None, model_update_ver, False, False]) # noqa :E203

                micro_idx[dev_id] += 1
                is_target[dev_id] += 1
            else:
                block = None
            sched.append(PipelineScheduleKD(block) if block is not None else None)
        schedule.append(sched)
    
    grad_micro_idx = [num_ubatch - 1 for _ in range(num_devices)]
    backward_start = num_devices - 1

    # Backward
    for bat_id in range(num_ubatch + (num_devices - 1)):
        sched = []
        for dev_id in range(num_devices):
            prev = dev_id + 1 if dev_id < num_devices - 1 else None

            if dev_id >= backward_start and grad_micro_idx[dev_id] >= 0:
                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               0, None, None, True, False])
                grad_micro_idx[dev_id] -= 1
            elif grad_micro_idx[dev_id] == -1: # reach the end of the backward pass is optimization
                block = None
                grad_micro_idx[dev_id] -= 1
            else:
                block = None
            sched.append(PipelineScheduleKD(block) if block is not None else None)
        schedule.append(sched)
        backward_start -= 1
    
    # Optimization
    optim_sched = []
    for dev_id in range(num_devices):
        block = tuple([0, dev_id, None, False, batch_index, 0, None, None, False, True])
        optim_sched.append(PipelineScheduleKD(block))
    schedule.append(optim_sched)
    return schedule

def schedule_without_pipeline_kd(devices, batch_index, optimize_len, gpipe_num_ubatch=None) \
        -> Iterable[Iterable[Optional[PipelineScheduleKD]]]:
    # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)

    num_devices = devices
    num_ubatch = gpipe_num_ubatch if gpipe_num_ubatch is not None else num_devices - 1

    schedule = []
    micro_idx = [0 for _ in range(num_devices)]		# current microbatch index (0 to num_ubatch-1)
    is_target = [0 for _ in range(num_devices)]		# Count of microbatches processed to determine if view is teacher/student. from 0 to 2*num_ubatch each value
    forward = [True for _ in range(num_devices)]	# forward active flag
    
    # Forward
    for bat_id in range(2 * num_ubatch + (num_devices - 1)):
        sched = []
        for dev_id in range(num_devices):
            if micro_idx[dev_id] >= num_ubatch:
                micro_idx[dev_id] -= num_ubatch # keeps micro index in range (0 to num_ubatch-1)
                
            prev = dev_id - 1 if dev_id > 0 else None
            
            if is_target[dev_id] >= num_ubatch * 2:
                is_target[dev_id] -= 2 * num_ubatch
                forward[dev_id] = False
            is_target_bool = is_target[dev_id] < num_ubatch
            
            if micro_idx[dev_id] == 0 and is_target[dev_id] == 0:
                model_update_ver = batch_index - 1
            else:
                model_update_ver = None
            if dev_id <= bat_id and forward[dev_id]:
                #            (i,         j,     prev_partition, is_target,      batch_idx,   view_idx, dependencies, model_update_ver, backprop, optimize) # noqa :E203
                block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index, 0, None, model_update_ver, False, False]) # noqa :E203

                micro_idx[dev_id] += 1
                is_target[dev_id] += 1
            else:
                block = None
            sched.append(PipelineScheduleKD(block) if block is not None else None)
        schedule.append(sched)
    
    grad_micro_idx = [num_ubatch - 1 for _ in range(num_devices)]
    backward_start = num_devices - 1

    # Backward
    for bat_id in range(num_ubatch + (num_devices - 1) + 1):
        sched = []
        for dev_id in range(num_devices):
            prev = dev_id + 1 if dev_id < num_devices - 1 else None

            if dev_id >= backward_start and grad_micro_idx[dev_id] >= 0:
                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               0, None, None, True, False])
                grad_micro_idx[dev_id] -= 1
            elif grad_micro_idx[dev_id] == -1: # reach the end of the backward pass is optimization
                block = tuple([0, dev_id, None, False, batch_index, 0, None, None, False, True])
                grad_micro_idx[dev_id] -= 1
            else:
                block = None
            sched.append(PipelineScheduleKD(block) if block is not None else None)
        schedule.append(sched)
        backward_start -= 1
    return schedule

def schedule_start_kd(devices, batch_index, micro_idx, is_target, view_idx, optimize_len, skip=False, num_ubatch=6) \
        -> Iterable[Iterable[Optional[PipelineScheduleKD]]]:
    num_devices = devices
    # num_ubatch = num_devices - 1
    start = []
    for bat_id in range(num_ubatch):
        schedule = []
        for dev_id in range(num_devices):

            if micro_idx[dev_id] >= num_ubatch:
                micro_idx[dev_id] -= num_ubatch

            prev = dev_id - 1 if dev_id > 0 else None

            if is_target[dev_id] >= num_ubatch * 2:
                is_target[dev_id] -= num_ubatch * 2

            if dev_id <= bat_id:
                block = tuple([micro_idx[dev_id], dev_id, prev, True, batch_index, 0, None,
                               None, False, False])
                micro_idx[dev_id] = micro_idx[dev_id] + 1
                # view_idx[dev_id] = view_idx[dev_id] + 1
                is_target[dev_id] = is_target[dev_id] + 1
            else:
                block = None
            schedule.append(PipelineScheduleKD(block) if block is not None else None)
        start.append(schedule)
    return start

def new_schedule_start_kd(devices, batch_index, micro_idx, is_target, view_idx, 
                          optimize, batch_count, optimize_len, num_ubatch=6):
    num_devices = devices
    repeat = []

    backprop = [-1 for _ in range(num_devices)]
    propagate = [0 for _ in range(num_devices)]
    grad_micro_idx = [2 * num_ubatch - 1 for _ in range(num_devices)]

    for _ in range(4 * num_ubatch + optimize_len):
        schedule = []
        dependency = None
        
        for dev_id in range(num_devices):
            if backprop[dev_id] == -1: # Forward 단계 스케줄링 (해당 파티션이 아직 backward를 시작하지 않은 경우)

                if optimize[dev_id] > 0:
                    if optimize[dev_id] == optimize_len:
                        block = tuple([0, dev_id, 0, False, batch_index, 0, None, None, False, True])
                    optimize[dev_id] -= 1
                else:
                    if micro_idx[dev_id] >= num_ubatch:
                        micro_idx[dev_id] -= num_ubatch

                    prev = dev_id - 1 if dev_id > 0 else None

                    if is_target[dev_id] >= num_ubatch * 2:
                        is_target[dev_id] -= num_ubatch * 2
                    is_target_bool = is_target[dev_id] < num_ubatch

                    # Dependency. 마지막 파티션이 마지막 forward를 처리할 경우 dependency 설정 
                    if dev_id == num_devices - 1 and is_target[dev_id] == 2*num_ubatch - 1:
                        dependency = batch_index
                        backprop[dev_id] = 2*num_ubatch - 1 # (나중 backward 시작 시점)

                    # Model Update Version (tag the model state that was used for this forward pass )
                    if micro_idx[dev_id] == 0 and batch_count[dev_id] % num_ubatch == 0: # first microbatch of forward and start of the full pipeline cycle
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None

                    if batch_count[dev_id] >= num_ubatch:
                		#            (i,         j,     prev_partition, is_target,      batch_idx,   
                		#   	view_idx, dependencies, model_update_ver, backprop, optimize) # noqa :E203
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index + 1,
                                       0, dependency, model_update_ver, False, False])
                    else:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index,
                                       0, dependency, model_update_ver, False, False])

                    micro_idx[dev_id] += 1
                    is_target[dev_id] += 1
                    batch_count[dev_id] += 1

                    if batch_count[dev_id] >= 2*num_ubatch and optimize[dev_id] == 0:
                        optimize[dev_id] = optimize_len
                        batch_count[dev_id] = 0
            else: # Backward 단계 스케줄링 (해당 파티션이 backward 단계에 들어간 경우)
                if grad_micro_idx[dev_id] < 0:
                    grad_micro_idx[dev_id] += num_ubatch

                prev = dev_id + 1 if dev_id < num_devices - 1 else None

                if dev_id > 0 and propagate[dev_id] < 1:
                    propagate[dev_id] += 1
                    if propagate[dev_id] == 1:
                        backprop[dev_id - 1] = 2*num_ubatch - 1

                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               0, None, None, True, False])

                backprop[dev_id] -= 1
                grad_micro_idx[dev_id] -= 1

            schedule.append(PipelineScheduleKD(block) if block is not None else None)
        repeat.append(schedule)
    return repeat # -> List[List[PipelineScheduleKD]]바깥 리스트: 각 clock (시간 스텝)

def new_schedule_repeat_kd(devices, batch_index, micro_idx, is_target, view_idx,
                        optimize, batch_count, optimize_len, num_ubatch=6):
    num_devices = devices
    repeat = []

    backprop = [-1 for _ in range(num_devices)]
    propagate = [0 for _ in range(num_devices)]
    grad_micro_idx = [2 * num_ubatch - 1 for i in range(num_devices)]
    reset = False
    
    for _ in range(4 * num_ubatch + optimize_len):
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
                    if micro_idx[dev_id] >= num_ubatch:
                        micro_idx[dev_id] -= num_ubatch
                        if dev_id == num_devices - 1:
                            reset = True

                    # previous Partitions
                    prev = dev_id - 1 if dev_id > 0 else None

                    if is_target[dev_id] >= num_ubatch * 2:
                        is_target[dev_id] -= num_ubatch * 2
                    is_target_bool = is_target[dev_id] < num_ubatch

                    # Dependency
                    if dev_id == num_devices - 1 and is_target[dev_id] == 2 * num_ubatch - 1:
                        dependency = batch_index
                        backprop[dev_id] = 2*num_ubatch - 1

                    # Model Update Ver
                    if micro_idx[dev_id] == 0 and batch_count[dev_id] % num_ubatch == 0:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None

                    if batch_count[dev_id] >= num_ubatch and reset:
                		#            (i,         j,     prev_partition, is_target,      batch_idx,   
                		#   	view_idx, dependencies, model_update_ver, backprop, optimize)
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index + 1,
                                       0, dependency, model_update_ver, False, False])
                    else:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index,
                                       0, dependency, model_update_ver, False, False])

                    micro_idx[dev_id] += 1
                    is_target[dev_id] += 1
                    batch_count[dev_id] += 1

                    if batch_count[dev_id] >= 2*num_ubatch and optimize[dev_id] == 0:
                        optimize[dev_id] = optimize_len
                        batch_count[dev_id] = 0
            else:
                if grad_micro_idx[dev_id] < 0:
                    grad_micro_idx[dev_id] += num_ubatch

                prev = dev_id + 1 if dev_id < num_devices - 1 else None
                
                if dev_id > 0 and propagate[dev_id] < 1:
                    propagate[dev_id] += 1
                    if propagate[dev_id] == 1:
                        backprop[dev_id - 1] = 2*num_ubatch - 1

                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               0, None, None, True, False])

                backprop[dev_id] -= 1
                grad_micro_idx[dev_id] -= 1

            schedule.append(PipelineScheduleKD(block) if block is not None else None)
        repeat.append(schedule)
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


class TSPipeSchedulerKD():
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

    # 스케줄 내 작업 유효성 체크: 스케줄에서 요구하는 batch가 input queue에 들어올 때까지 blocking
    def throttle_until_batch_available(self, lst_sched: Iterable[Optional[PipelineScheduleKD]]):
        # throttle scheduling
        for subsched in lst_sched:
            if subsched is not None:
                self.input_batch_queue.wait_batch(subsched.batch_idx)

    @sanity_check
    def schedule_generator(self) -> Generator[List[Optional[PipelineScheduleKD]], None, None]:
        """Generates schedules for each clock cycle."""
        # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)
        n = num_devices = self.num_partitions
        optimize_len = 1

        schedule = None
        if self.gpipe_emulation_enabled:
            batch_idx = 0
            while True:
                batch_idx += 1
                schedule = schedule_without_pipeline_kd(n, batch_idx, optimize_len, gpipe_num_ubatch=self.gpipe_num_ubatch)
                for sched in schedule:
                    self.throttle_until_batch_available(sched)
                    yield sched

        # 초기 warm-up 단계 (Staleness 허용)
        for batch_idx in range(1, self.num_skip_staleness + 1):
            schedule = schedule_without_pipeline_kd(n, batch_idx, optimize_len)
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
        schedule = schedule_start_kd(n, batch_index, micro_idx, is_target, view_idx, optimize_len)
        for sched in schedule:
            self.throttle_until_batch_available(sched)
            yield sched
        
        # forward + backward 초기화 (시동)
        schedule = new_schedule_start_kd(n, batch_index, micro_idx, is_target, view_idx,
                                      optimize, batch_count, optimize_len)
        for sched in schedule:
            self.throttle_until_batch_available(sched)
            yield sched
        batch_index += 1

        # 본격적으로 pipeline 학습 반복 시작. 매 cycle에 대해 파티션별로 forward/backward 작업 할당
        while True:
            schedule = new_schedule_repeat_kd(n, batch_index, micro_idx, is_target, view_idx,
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
        schedule = schedule_without_pipeline_kd(n, batch_idx)
    print()
    micro_idx = [0 for i in range(n)]
    view_idx = [0 for i in range(n)]
    is_target = [0 for i in range(n)]
    optimize = [0 for i in range(n)]

    schedule = schedule_start_kd(n, 3, micro_idx, is_target, view_idx)
    print()
    schedule = new_schedule_start_kd(n, 3, micro_idx, is_target, view_idx, optimize)
    print()
    schedule = new_schedule_repeat_kd(n, 4, micro_idx, is_target, view_idx, optimize)
    print()
    schedule = new_schedule_repeat_kd(n, 5, micro_idx, is_target, view_idx, optimize)

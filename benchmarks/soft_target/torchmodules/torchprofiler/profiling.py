# Modified for TSPipe by Eunjin Lee

import torch
import torch.nn as nn
import os
import time
import sys
from collections import OrderedDict

class Profiling(object):
    def __init__(self, model, module_whitelist):
        if isinstance(model, torch.nn.Module) is False:
            raise Exception("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.module_whitelist = module_whitelist
        self.profiling_on = True
        self.hook_done = False

        self.forward_event_dict = OrderedDict()
        self.backward_event_dict = OrderedDict()
        self.forward_dict = OrderedDict()
        self.backward_dict = OrderedDict()
        
        self.handles = []
        self.accumulated_activation_size_bytes = OrderedDict() # Parent module: accumulated activation size, key: parent_id

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def processed_results(self):
        records = []
        for index, (module, forward_info) in enumerate(self.forward_dict.items()):
            module_name = str(module)
            _, forward_time_ms, input_act_shape, output_act_shape, param_size_bytes \
                                                                    = forward_info
            accum_activation_size_bytes = self.accumulated_activation_size_bytes[str(index)]
            
            if module in self.backward_dict:
                _, backward_time_ms = self.backward_dict[module]
                records.append((
                    module_name,
                    forward_time_ms,                                             # Convert to ms
                    backward_time_ms if backward_time_ms is not None else -1,
                    input_act_shape,
                    output_act_shape,
                    param_size_bytes,
                    accum_activation_size_bytes  # Convert to bytes
                ))
            else:
                records.append((
                    module_name,
                    forward_time_ms,                                          
                    -1,
                    input_act_shape,
                    output_act_shape,
                    param_size_bytes,
                    accum_activation_size_bytes
                ))

        return records
    
    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.register_hooks(self.model)
        self.profiling_on = True
        return self

    def stop(self):
        self.profiling_on = False
        self.reset_hooks()
        return self
        
    def reset_hooks(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        
    # =============================
    # Forward and Backward hooks    
    # =============================
    def _forward_pre_hook(self, module, input):
        if not self.profiling_on:
            return

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.forward_event_dict[module] = (start_event, end_event)

    def _forward_hook(self, module, input, output):
        if not self.profiling_on or module not in self.forward_event_dict:
            return
        
        start_event, end_event = self.forward_event_dict[module]
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

        input_shape = list(input[0].size()) if isinstance(input, tuple) and len(input) > 0 else None
        output_shape = list(output.size()) if isinstance(output, torch.Tensor) else \
                    list(output[0].size()) if isinstance(output, (list, tuple)) and output else None
        param_size_bytes = sum(p.numel() * p.element_size() for p in module.parameters() if p.requires_grad)

        self.forward_dict[module] = [os.getpid(), elapsed_time_ms, input_shape, output_shape, param_size_bytes]

    def _backward_pre_hook(self, module, grad_output):
        if not self.profiling_on:
            return grad_output
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.backward_event_dict[module] = (start_event, end_event)
        return tuple(g.clone() for g in grad_output)

    def _backward_hook(self, module, grad_input, grad_output):
        if not self.profiling_on or module not in self.backward_event_dict:
            return grad_input

        start_event, end_event = self.backward_event_dict[module]
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)  # float(ms)
        self.backward_dict[module] = [os.getpid(), elapsed_time_ms]
        return tuple(g.clone() if g is not None else None for g in grad_input)

    def _forward_hook_deep_wrapper(self, parent_id):
        def forward_hook_deep(module, input, output):
            if isinstance(output, torch.Tensor):
                act_size_bytes = output.numel() * output.element_size()
            elif isinstance(output, (list, tuple)):
                act_size_bytes = sum(o.numel() * o.element_size() for o in output if isinstance(o, torch.Tensor))
            else:
                act_size_bytes = 0

            self.accumulated_activation_size_bytes.setdefault(parent_id, 0)
            self.accumulated_activation_size_bytes[parent_id] += act_size_bytes
        return forward_hook_deep

    def register_deep_hooks(self, parent_id, module):
        if len(list(module.children())) == 0: # register only leaf modules
            h5 = module.register_forward_hook(self._forward_hook_deep_wrapper(parent_id))
            self.handles.append(h5)
        else:
            for child_name, child_module in module.named_children():
                self.register_deep_hooks(parent_id, child_module)

    def register_hooks(self, model):
        self.reset_hooks()
        for name, module in model.named_children():              
            """
                We assume the model is already in a nn.Sequential format.
                See line 206 in /benchmarks/soft_target/train_kd.py in TSPipe
            """
            handle1 = module.register_forward_pre_hook(self._forward_pre_hook)
            handle2 = module.register_forward_hook(self._forward_hook)
            handle3 = module.register_full_backward_pre_hook(self._backward_pre_hook)
            handle4 = module.register_full_backward_hook(self._backward_hook)
            self.handles.extend([handle1, handle2, handle3, handle4])

            # Register deep hooks for accumulated activation size.
            self.register_deep_hooks(name, module)
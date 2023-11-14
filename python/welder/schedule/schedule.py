from typing import Dict, List

from tvm import te

from ..config import Config, Stride
from ..te_utils import get_compute_ops, seperate_reduce_ops
from .te_elementwise import *
from .tir_elementwise import *
from .te_reduce import *
from .te_reduce_interthread import *
from .te_wmma import *
from .tir_mma import TIRCutlassMMAScheduler
from .tir_simt import *
from .tir_reduce_interthread import *
from .tir_ladder import TIRLadderMMAScheduler4D
from .tir_ladder_pad import TIRLadderMMAPadScheduler2D
import logging 

logger = logging.getLogger(__name__)

def schedule(args: List[te.Tensor], config: Config, shared_inputs: List[te.Tensor] = [],
        shared_inputs_strides: Dict[te.Tensor, Stride] = {}, shared_outputs = []):
    
    input_args, output_args = [], []
    for arg in args:
        if isinstance(arg.op, te.PlaceholderOp):
            input_args.append(arg)
        else:
            output_args.append(arg)
    
    ops = get_compute_ops(args)
    reduces_ops, _ = seperate_reduce_ops(ops)
    schedule_on_inner_stage = False
    for tensor in args:
        if isinstance(tensor.op, te.ComputeOp) and tensor.name not in config.schedule_stages:
            schedule_on_inner_stage = True

    if len(reduces_ops) == 0:
        assert(not schedule_on_inner_stage)
        template = TIRElementWiseScheduler if len(output_args) == 1 else TEElementWiseScheduler
    elif config.use_ladder:
        if not config.use_tc:
            template = TIRSIMTScheduler
        elif len(config.rstep) == 1:
            template = TIRLadderMMAPadScheduler2D
        elif len(config.rstep) == 2:
            template = TIRLadderMMAScheduler4D
        else:
            raise NotImplementedError("Schedule not implemented")
    elif config.use_tc and config.use_cutlass:
        template = TIRCutlassMMAScheduler
    elif config.use_tc and not config.use_cutlass:
        if schedule_on_inner_stage: raise NotImplementedError("Schedule not implemented")
        template = TEWarpMMAScheduler
    elif any([t > 1 for t in config.reduce_thread]):
        if schedule_on_inner_stage: raise NotImplementedError("Schedule not implemented")
        template = TIRReduceInterThreadScheduler
    else:
        template = TIRSIMTScheduler

    logger.debug(f"Using template: {template} config: {config}")

    def initialize_scheduler(template, args, config, shared_inputs, shared_outputs, shared_inputs_strides):
        scheduler = template(args, config)
        scheduler.shared_inputs = shared_inputs
        scheduler.shared_outputs = shared_outputs
        scheduler.shared_inputs_strides = {tensor: Stride() for tensor in shared_inputs}
        scheduler.shared_inputs_strides.update(shared_inputs_strides)
        scheduler.make_passes()
        scheduler.schedule()
        return scheduler

    if template == TIRElementWiseScheduler:
        try:
            scheduler = initialize_scheduler(template, args, config, shared_inputs, shared_outputs, shared_inputs_strides)
        except Exception as e:
            logger.debug(f"Tir template failed because {e}, fallback to te")
            template = TEElementWiseScheduler
            scheduler = initialize_scheduler(template, args, config, shared_inputs, shared_outputs, shared_inputs_strides)
    elif template == TIRReduceInterThreadScheduler:
        try:
            scheduler = initialize_scheduler(template, args, config, shared_inputs, shared_outputs, shared_inputs_strides)
        except Exception as e:
            if any([t > 1 for t in config.reduce_thread]) and not schedule_on_inner_stage:
                logger.debug(f"Tir template failed because {e}, fallback to te")
                template = TEReduceInterThreadScheduler
                scheduler = initialize_scheduler(template, args, config, shared_inputs, shared_outputs, shared_inputs_strides)
    else:
        scheduler = initialize_scheduler(template, args, config, shared_inputs, shared_outputs, shared_inputs_strides)

    

    return scheduler

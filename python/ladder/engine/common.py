import json
from typing import List
import os
from ..graph import Node, PlaceHolderNode
from ..utils import CompileResult


def get_id_by_name(name):
    return int(name.split("_")[-1])


class FusionGroup:
    def __init__(
        self,
        node_list: List[Node],
        group_id: int,
        cpresult: "CompileResult",
        gain: float,
    ) -> None:
        self.nodes = node_list
        self.group_id = group_id
        self.cpresult = cpresult
        self.gain = gain


def dump(fusion_groups: List[FusionGroup]):
    obj = []
    result_reuse_map = {}
    for group in fusion_groups:
        group_desc = {}
        node_names = [node.name for node in group.nodes]
        group_desc["nodes"] = [get_id_by_name(name) for name in node_names]
        group_desc["node_names"] = node_names
        group_desc["group_id"] = group.group_id
        if group.cpresult is not None:
            cpresult = group.cpresult
            group_desc["input_desc"] = [
                [name, get_id_by_name(name), id] for name, id in cpresult.input_desc
            ]
            group_desc["output_desc"] = [
                [name, get_id_by_name(name), id] for name, id in cpresult.output_desc
            ]

            if cpresult.origin in result_reuse_map:
                cpresult = result_reuse_map[cpresult.origin]
            else:
                result_reuse_map[cpresult.origin] = cpresult
            group_desc["code"] = cpresult.code
            group_desc["block_size"] = [int(it) for it in cpresult.block_size]
            group_desc["grid_size"] = [int(it) for it in cpresult.grid_size]
            group_desc["latency"] = cpresult.latency
            group_desc["name"] = cpresult.name
            group_desc["gain"] = group.gain
        obj.append(group_desc)
    return obj


def save_models(ordered_nodes, fname: str):
    model_infos = []
    def filter_dtypes(dtypes):
        return [str(dtype) for dtype in dtypes]
    
    for idx, node in enumerate(ordered_nodes):
        model_info = {
            "idx": idx,
            "name": node.name,
        }
        # include input_names, input_shapes, input_dtypes, and input_id?
        inputs = []
        for edge in node._in_edges:
            src = edge.src_node
            if isinstance(src, PlaceHolderNode):
                # it's input node
                _input_info = {
                    "name": src.name,
                    "shapes": src._shapes,
                    "dtypes": filter_dtypes(src._dtypes),
                }
                inputs.append(_input_info)
        model_info["inputs"] = inputs
        # outputs
        outputs = []
        for edge in node._out_edges:
            dst = edge.dst_node
            _output_info = {
                "name": dst.name,
                "shapes": dst._shapes,
                "dtypes": filter_dtypes(dst._dtypes),
            }
            outputs.append(_output_info)
        model_info["outputs"] = outputs
        model_infos.append(model_info)
    with open(fname, "w") as f:
        json.dump(model_infos, f, indent=2)
    return None


def save_results(fusion_groups: List[FusionGroup], fname: str):
    obj = dump(fusion_groups)
    with open(fname, "w") as f:
        json.dump(obj, f, indent=2)
    return None


def export_groups(fusion_groups: List[FusionGroup], directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    obj = dump(fusion_groups)

    for i, fgroup in enumerate(fusion_groups):
        _group_desc = obj[i]
        group_name = (
            f"group_{_group_desc['group_id']}_{'_'.join(_group_desc['node_names'])}"
        )
        # create a directory for each group
        group_name = group_name[:32]
        # clip the group name
        group_dir = os.path.join(directory, group_name)

        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        # save kernel code
        fname = f"{group_name}.cu"

        with open(os.path.join(group_dir, fname), "w") as f:
            if not hasattr(_group_desc, "code"):
                continue
            code = _group_desc["code"]
            comments = f"""
// group_id: {_group_desc['group_id']}
// node_names: {_group_desc['node_names']}
// gain: {_group_desc['gain']}
// latency: {_group_desc['latency']}
// grid_size: dim3({_group_desc['grid_size'][0]}, {_group_desc['grid_size'][1]}, {_group_desc['grid_size'][2]})
// block_size: dim3({_group_desc['block_size'][0]}, {_group_desc['block_size'][1]}, {_group_desc['block_size'][2]})
"""
            code = comments + code
            f.write(code)

        # save schedule_mods
        for op_name, mod_script in fgroup.cpresult.scheduled_mods.items():
            fname = f"{op_name}.py"
            with open(os.path.join(group_dir, fname), "w") as f:
                f.write(mod_script)

    return None


__all__ = ["save_results", "FusionGroup", "export_groups"]

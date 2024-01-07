import json
from typing import List

from ..graph import IRNode, Node, OutputNode
from ..lang import translate_ir_to_tvm

# internal name for debug
def get_node_name(id, op_type):
    return "_".join([op_type, str(id)])

def load_model(fname: str) -> List[Node]:
    with open(fname) as f:
        a = json.load(f)

    node_map = {item[0] : None for item in a}
    ordered_nodes = []
    for node_id, ir, op_type, inputs in a:
        anno, options = ir.find('## @'), []
        if anno >= 0:
            ir, options = ir[:anno].strip(), ir[ir.index(':', anno) + 1:].strip().split('|')
            options = [option.strip() for option in options]
        input_list = []
        for src_node, src_id in inputs:
            if src_node not in node_map:
                input_list.append(None)
            else:
                assert node_map[src_node] is not None, "Detected ring in topo order {}->{} !".format(src_node, node_id)
                input_list.append([node_map[src_node], src_id])
        if op_type == "Result":
            if input_list[0] is not None:
                node = OutputNode(*input_list[0])
            else:
                # result node connect to placeholder, simply ignore
                continue
        else:
            option_kv = {}
            for option in options:
                index = option.find("=")
                if index > 0:
                    option_kv[option[0:index]] = eval(option[index+1:])
                else:
                    option_kv[option] = True
            if "skip" in option_kv and option_kv["skip"] == True:
                args = None
            else:
                input_args, output_args = translate_ir_to_tvm(ir)
                args = input_args + output_args
                input_idx_list = []
                for arg in input_args:
                    assert arg.name.startswith("input")
                    input_idx = int(arg.name[5:])
                    input_idx_list.append(input_idx)
                new_input_list = [input_list[i] for i in input_idx_list]
                if input_list != new_input_list:
                    option_kv["nnf_input_idx"] = input_idx_list
                    input_list = new_input_list
            node = IRNode(input_list, args, get_node_name(node_id, op_type))
            for k, v in option_kv.items(): node.add_tag(k, v)
        node_map[node_id] = node
        ordered_nodes.append(node)
    # check for tensorCore shapes
    for node in ordered_nodes:
        if node.get_tag("skip"): continue
        if node.get_tag("tensorCoreConfig"):
            C_ax_m, C_ax_n = node.get_tag("tensorCoreConfig")
            shape = node.get_shape()
            raxis_invalid = any([r % 16 != 0 for r in node.raxis.values()])
            if shape[C_ax_m] % 8 != 0 or shape[C_ax_n] % 8 != 0 or shape[C_ax_m] * shape[C_ax_n] % 256 != 0 \
                or raxis_invalid:
                node.add_tag("tensorCoreConfig", False)
    return ordered_nodes

__all__ = ['load_model']

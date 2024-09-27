# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from typing import List
import tvm
from tvm.contrib.popen_pool import PopenPoolExecutor

from ..code_generator import CodeGenerator
from ..graph import find_topo_sort, IRNode, PlaceHolderNode, OutputNode
from ..utils import CompileResult
from .base_tunner import Tunner, _extract_subgraph, eliminate_memcpy


class _save:
    pass


def init_server(data_list):
    ordered_nodes = load_tile_graph(data_list)
    _save.node_map = {node.name: node for node in ordered_nodes}


def call_build(node_names, connections, send_config, kernel_name, target_str):
    cgen = CodeGenerator()
    nodes = [_save.node_map[name] for name in node_names]
    output_nodes, _, _ = _extract_subgraph(nodes, connections)
    eliminate_memcpy(output_nodes)
    config = {}
    config["globals"] = send_config["globals"]
    for node in find_topo_sort(output_nodes):
        if node.name in send_config:
            config[node] = send_config[node.name]
    cpresult = cgen.compile(
        output_nodes, config, tvm.target.Target(target_str), kernel_name=kernel_name
    )
    return [
        cpresult.code,
        cpresult.block_size,
        cpresult.grid_size,
        cpresult.args,
        cpresult.scheduled_mods,
        cpresult.arg_op_mapping_list
    ]


def save_tile_graph(ordered_nodes: List[IRNode]) -> List:
    all_nodes = find_topo_sort(ordered_nodes)
    data_list = []
    idx_map = {}
    for i, node in enumerate(all_nodes):
        idx_map[node] = i
        tags = node._tag
        node_type = type(node)
        name = node.name
        ir = node.get_ir()
        edges = [(idx_map[edge.src_node], edge.src_id) for edge in node.inputs]
        data_list.append((node_type, name, ir, tags, edges))
    return data_list


def load_tile_graph(data_list: List):
    idx_map = {}
    ordered_nodes = []
    for i, data in enumerate(data_list):
        node_type, name, ir, tags, edges = data
        inputs = [(idx_map[node_id], output_id) for node_id, output_id in edges]
        if node_type == IRNode:
            node = IRNode(inputs, tvm.ir.load_json(ir), name)
        elif node_type == PlaceHolderNode:
            node = PlaceHolderNode(name)
        elif node_type == OutputNode:
            node = OutputNode(*inputs[0])
        for k, v in tags.items():
            node.add_tag(k, v)
        idx_map[i] = node
        ordered_nodes.append(node)
    return ordered_nodes


class MultiProcTunner(Tunner):
    def __init__(
        self, ordered_nodes: List[IRNode], arch, device="cuda:0", check=False, topk=10
    ) -> None:
        super().__init__(arch, device, check, topk)
        num_procs = min(topk, os.cpu_count(), 10)
        self.pool = PopenPoolExecutor(
            max_workers=num_procs,
            timeout=20,
            initializer=init_server,
            initargs=[save_tile_graph(ordered_nodes)],
        )

    def generate_code(self, output_nodes, configs, kernel_name):
        compile_results = []
        node_names = [node.name for node in self.current_nodes]
        futures = []
        for config in configs:
            send_config = {
                key.name if isinstance(key, IRNode) else key: config[key]
                for key in config
            }
            send_config["globals"] = config["globals"]
            futures.append(
                self.pool.submit(
                    call_build,
                    node_names,
                    self.local_connections,
                    send_config,
                    kernel_name,
                    str(self.arch.target),
                )
            )
        for future, config in zip(futures, configs):
            try:
                result = future.result()
            except TimeoutError as e:
                self.write_error_log(config, "Compiler timeout.")
                continue
            except Exception as e:
                self.write_error_log(config, e)
                continue
            code, block_size, grid_size, args, mods, arg_op_mapping_list = result
            temp_cpresult = CompileResult(
                config,
                code,
                block_size,
                grid_size,
                kernel_name,
                args,
                scheduled_mods=mods,
            )
            # compile_results.append(
            #     CompileResult(
            #         config,
            #         code,
            #         block_size,
            #         grid_size,
            #         kernel_name,
            #         args,
            #         scheduled_mods=mods,
            #     )
            # )
            temp_cpresult.arg_op_mapping_list=arg_op_mapping_list
            compile_results.append(temp_cpresult)


        return compile_results

import hashlib
import logging
import numpy as np

from ..code_generator import CodeGenerator
from ..graph import Edge, OutputNode, find_topo_sort
from ..policy import DefaultPolicy, TCPolicy, LadderPolicy
from ..reference import get_subgraph_reference_outputs
from ..utils import CompileResult, compile_parallel
from .utils import insert_local_connections
from . import profiler
from tvm.contrib.popen_pool import PopenPoolExecutor

logger = logging.getLogger(__name__)

def get_max_diff(tensor_list_a, tensor_list_b):
    assert len(tensor_list_a) > 0
    total_diff = [0]
    for a, b in zip(tensor_list_a, tensor_list_b):
        assert a.shape == b.shape
        if np.any(np.logical_and(np.isnan(a), np.logical_not(np.isnan(b)))):
            return 1e7
        diff = np.abs(a-b)
        diff = diff / np.abs(b).clip(1) # handle large floating numbers
        diff = np.max(diff)
        if a.dtype == np.float16:
            diff /= 1000.0
        total_diff.append(diff)
    total_diff = max(total_diff)
    return total_diff

def subgraph_hash(output_nodes):
    nodes = find_topo_sort(output_nodes)
    node_hash, edge_hash = [], []
    node_idx = {node : i for i, node in enumerate(nodes)}
    for node in nodes:
        code = node.get_ir()
        hex = hashlib.sha1(bytes(code, encoding="utf-8")).hexdigest()
        node_hash.append(int(hex, 16))
        for edge in node.outputs:
            sig = (node_idx[edge.src_node], node_idx[edge.dst_node], edge.src_id, edge.dst_id)
            edge_hash.append(hash(sig))
    graph_sig = hash(tuple(node_hash + edge_hash))
    return graph_sig

def _extract_subgraph(nodes, connections=[]):
    node_map = {}
    output_nodes = []
    for node in nodes:
        input_list = []
        for edge in node.inputs:
            if edge.src_node in nodes:
                input_list.append((node_map[edge.src_node], edge.src_id))
            else:
                input_list.append(None)
        node_map[node] = node.clone(input_list)
        global_output = set()
        for edge in node.outputs:
            if not edge.dst_node in nodes:
                global_output.add(edge.src_id)
        for idx in global_output:
            output_nodes.append(OutputNode(node_map[node] ,idx))

    # get subgraph inputs and outputs mapping to the original nodes
    # this should get the same results as final code generation
    ordered_nodes = find_topo_sort(output_nodes)
    input_desc, output_desc = [], []
    for node in ordered_nodes:
        if node.is_placeholder():
            dst_node = node.outputs[0].dst_node
            dst_id = node.outputs[0].dst_id
            if dst_node.get_tag("nnf_input_idx"):
                dst_id = dst_node.get_tag("nnf_input_idx")[dst_id]
            input_desc.append([dst_node.name, dst_id])
        elif node.is_output():
            output_desc.append([node.inputs[0].src_node.name, node.inputs[0].src_id])

    insert_local_connections(output_nodes, connections)

    # sort input_desc and output_desc by new_ordered_nodes
    new_ordered_nodes = find_topo_sort(output_nodes)
    nodes_old = list(filter(lambda n: n.is_placeholder(), ordered_nodes))
    nodes_new = list(filter(lambda n: n.is_placeholder(), new_ordered_nodes))
    input_desc = [input_desc[nodes_old.index(nodes_new[i])] for i in range(len(input_desc))]
    nodes_old = list(filter(lambda n: n.is_output(), ordered_nodes))
    nodes_new = list(filter(lambda n: n.is_output(), new_ordered_nodes))
    output_desc = [output_desc[nodes_old.index(nodes_new[i])] for i in range(len(output_desc))]

    return output_nodes, input_desc, output_desc

def eliminate_memcpy(output_nodes):
    nodes = find_topo_sort(output_nodes)
    eliminated_node_cnt = 0
    for node in nodes:
        if node.get_tag("memcpy") and len(node.inputs) == 1 and len(node.outputs) == 1:
            inode = node.inputs[0].src_node
            onode = node.outputs[0].dst_node
            inode_id = node.inputs[0].src_id
            onode_id = node.outputs[0].dst_id
            if inode.is_placeholder() and not onode.is_output():
                inode.set_shape(node.get_shape(), overwrite=True)
                edge = Edge(inode, onode, inode_id, onode_id)
                inode.set_outputs(inode_id, edge)
                onode.set_inputs(onode_id, edge)
                eliminated_node_cnt += 1
            elif not inode.is_placeholder() and onode.is_output():
                if sum([edge.src_id == inode_id for edge in inode.outputs]) > 1: continue
                onode.set_shape(inode.get_shape(inode_id), overwrite=True)
                edge = Edge(inode, onode, inode_id, onode_id)
                inode.set_outputs(inode_id, edge)
                onode.set_inputs(onode_id, edge)
                eliminated_node_cnt += 1
    if eliminated_node_cnt > 0:
        eliminate_memcpy(output_nodes)

class Tunner(object):
    def __init__(self, arch, device="cuda:0", check=False, topk=10) -> None:
        self._cache =  {}
        self.device = device
        self.check = check
        self.topk = topk
        self.arch = arch
        self.profiler = PopenPoolExecutor(max_workers=1, timeout=None, initializer=profiler.init_server, initargs=[arch])
        self.error_log = None

    def get_cache(self, sig):
        return self._cache[sig]

    def count_cache(self, sig):
        return sig in self._cache

    def dump_cache(self, fname: str):
        import pickle
        with open(fname, "wb") as f:
            pickle.dump(self._cache, f)

    def load_cache(self, fname: str):
        import pickle
        with open(fname, "rb") as f:
            self._cache = pickle.load(f)

    def set_cache(self, sig, value):
        self._cache[sig] = value

    def write_error_log(self, config, msg):
        if self.error_log is None:
            self.error_log = open("ladder_error.log", "w")
        print([node.name for node in self.current_nodes], file=self.error_log)
        print(config, file=self.error_log)
        print(msg, file=self.error_log, flush=True)

    def get_policy_list(self):
        policy_list = [DefaultPolicy]
        for node in self.current_nodes:
            if node.get_tag("tensorCoreConfig"):
                policy_list = [TCPolicy, DefaultPolicy]
                if node.get_tag("ladder_config"):
                    policy_list = [LadderPolicy]
                break
        return policy_list

    def generate_configs(self, policy_list, output_nodes):
        configs = []
        for policy in policy_list:
            remaining = self.topk - len(configs)
            if remaining <= 0:
                break
            configs.extend(policy(output_nodes, self.arch).emit_config(remaining))
        return configs

    def generate_code(self, output_nodes, configs, kernel_name):
        cgen = CodeGenerator()
        compile_results = []
        for config in configs:
            try:
                cpresult = cgen.compile(output_nodes, config, self.arch.target, kernel_name=kernel_name)
            except Exception as e:
                self.write_error_log(config, e)
                continue
            compile_results.append(cpresult)
        return compile_results

    def select_best(self, output_nodes, compile_results):
        for cpresult in compile_results:
            logger.debug(cpresult.config)
            if cpresult.lib_name is None:
                cpresult.latency = 1e8
            else:
                future = self.profiler.submit(profiler.call_profile, cpresult.lib_name, cpresult.args, self.device)
                try:
                    cpresult.latency = future.result()
                except ChildProcessError as e:
                    cpresult.latency = 1e8
                finally:
                    cpresult.remove_lib()
            logger.debug(cpresult.latency)
        compile_results = list(filter(lambda x:x.latency<1e8, compile_results))
        compile_results = sorted(compile_results, key=lambda x:x.latency)
        if len(compile_results) == 0:
            return None

        logger.debug(f"Best Config: {compile_results[0].config}")
        logger.info(f"result: {compile_results[0].latency}")
        if not self.check:
            return compile_results[0]

        ref_out = get_subgraph_reference_outputs(output_nodes, device=self.device)
        for best in compile_results:
            out = best.get_example_outputs(self.device)
            total_diff = get_max_diff(out, ref_out)
            logger.info(f"Diff: {total_diff}")
            if total_diff > 1e-3:
                continue
            return best
        return None
    
    def tune(self, nodes, local_connections=[], kernel_name="Fused"):
        if any([node.get_tag("skip") for node in nodes]):
            return None

        self.current_nodes = nodes
        self.local_connections = local_connections
        logger.info(f"Tuning {[node.name for node in self.current_nodes]}")
        output_nodes, input_desc, output_desc = _extract_subgraph(self.current_nodes, self.local_connections)
        if len(output_nodes) > 1:
            def _check_dims_is_the_same(shapes):
                # the check should be put under multiple output case.
                # check if all shapes are with the same dims
                # for relay check in type_solver.cc:236
                for shape in shapes:
                    if len(shape) != len(shapes[0]):
                        return False
                return True
            if not _check_dims_is_the_same([node.get_shape() for node in output_nodes]):
                return None
        eliminate_memcpy(output_nodes)
        signature = subgraph_hash(output_nodes)
        if self.count_cache(signature):
            logger.info("Found in cache")
            cached = self.get_cache(signature)
            if cached is None:
                return None
            result = CompileResult(cached.config, cached.code.replace(cached.name, kernel_name),
                cached.block_size, cached.grid_size, kernel_name, cached.args)
            result.latency = cached.latency
            result.set_io_desc(input_desc, output_desc)
            result.origin = cached
            return result

        policy_list = self.get_policy_list()

        try:
            configs = self.generate_configs(policy_list, output_nodes)
        except Exception as e:
            self.write_error_log(None, e)
            self.set_cache(signature, None)
            return None
        if len(configs) == 0:
            self.set_cache(signature, None)
            return None

        compile_results = self.generate_code(output_nodes, configs, kernel_name)
        for cpresult in compile_results:
            cpresult.set_io_desc(input_desc, output_desc)

        compile_parallel(compile_results, self.arch, timeout=30)
        best = self.select_best(output_nodes, compile_results)
        self.set_cache(signature, best)
        return best

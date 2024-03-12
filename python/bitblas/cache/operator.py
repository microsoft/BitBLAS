import bitblas
from bitblas.ops.operator import OperatorConfig, Operator
from dataclasses import asdict
import os
import json
import tempfile
from hashlib import sha256
import tvm
from tvm.contrib.tar import tar
import logging

logger = logging.getLogger(__name__)


class OperatorCache:
    """
    Manages a cache for operator instances (e.g., Matmul, Convolution) based on their configurations.
    """

    def __init__(self):
        self.cache = {}

    def add(self, config: OperatorConfig, op_inst: Operator):
        self.cache[config] = op_inst

    def get(self, config: OperatorConfig):
        return self.cache.get(config)

    def exists(self, config):
        return config in self.cache

    def clear(self):
        self.cache.clear()
    
    def size(self):
        return len(self.cache)

    def save_into_database(self, database_path=None, target=None):
        database_path = self._ensure_database_path(database_path)
        for config, op_inst in self.cache.items():
            arch_str = self._determine_arch_str(op_inst, target)
            arch_path = os.path.join(database_path, arch_str)
            self._ensure_directory(arch_path)
            hash_str = sha256(repr(config).encode()).hexdigest()
            config_path = os.path.join(arch_path, hash_str)
            self._ensure_directory(config_path)
            self._save_operator_config_and_artifact(config, op_inst, config_path)

    def load_from_database(self, database_path, target=None):
        if not os.path.exists(database_path):
            logger.error(f"Database path {database_path} does not exist")
            return
        arch_str = self._determine_target_arch_str(target)
        arch_path = os.path.join(database_path, arch_str)
        if not os.path.exists(arch_path):
            logger.error(f"Target {arch_str} does not exist in the database")
            return
        self._load_operators_from_arch_path(arch_path, target)

    def _ensure_database_path(self, database_path):
        if database_path is None:
            return tempfile.mkdtemp()
        os.makedirs(database_path, exist_ok=True)
        return database_path

    def _determine_arch_str(self, op_inst, target):
        return (
            target
            if target
            else "-".join(list(op_inst.target.keys) + [op_inst.target.arch])
        )

    def _ensure_directory(self, path):
        os.makedirs(path, exist_ok=True)

    def _save_operator_config_and_artifact(self, config, op_inst, config_path):
        config_type, operator_type = type(config).__name__, type(op_inst).__name__
        json.dump(
            asdict(config), open(os.path.join(config_path, f"{config_type}.json"), "w")
        )
        artifact_path = os.path.join(config_path, "tvm_rt_mod." + tar.output_format)
        op_inst.rt_mod.export_library(artifact_path, fcompile=tar)
        json.dump(
            {"config_type": config_type, "operator_type": operator_type},
            open(os.path.join(config_path, "mapping.json"), "w"),
        )

    def _determine_target_arch_str(self, target):
        return (
            target
            if isinstance(target, str)
            else "-".join(list(target.keys) + [target.arch])
        )

    def _load_operators_from_arch_path(self, arch_path, target):
        for root, dirs, _ in os.walk(arch_path):
            for directory in dirs:
                config_path = os.path.join(root, directory)
                self._load_operator(config_path, target)

    def _load_operator(self, config_path, target):
        mapping, config, rt_mod = None, None, None
        for file in os.listdir(config_path):
            full_path = os.path.join(config_path, file)
            if file == "mapping.json":
                mapping = json.load(open(full_path))
            elif file.endswith(".json"):
                config = json.load(open(full_path))
            elif file.endswith(".tar"):
                rt_mod = tvm.runtime.load_module(full_path)

        if mapping and config and rt_mod:
            self._instantiate_and_add_operator(mapping, config, rt_mod, target)

    def _instantiate_and_add_operator(self, mapping, config, rt_mod, target):
        config_cls = getattr(bitblas.ops, mapping["config_type"])
        operator_cls = getattr(bitblas.ops, mapping["operator_type"])
        op_inst = operator_cls(config=config_cls(**config), target=target)
        op_inst.update_runtime_module(rt_mod)
        self.add(config_cls(**config), op_inst)


global_operator_cache = OperatorCache()

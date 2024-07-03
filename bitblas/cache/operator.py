# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas.ops.operator import OperatorConfig, Operator
from dataclasses import asdict
import os
import json
import tempfile
from hashlib import sha256
import shutil
import tvm
from tvm.contrib.tar import tar
import logging

logger = logging.getLogger(__name__)

BITBLAS_DATABASE_PATH = os.path.expanduser("~/.cache/bitblas")


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
            # if the config already exists, skip saving
            if os.path.exists(config_path):
                continue
            self._ensure_directory(config_path)
            self._save_operator_config_and_artifact(config, op_inst, config_path)

    def load_from_database(self, database_path, target=None):
        if not os.path.exists(database_path):
            logger.info(
                f"Database path {database_path} does not exist, skipping loading operators from the database"
            )
            return
        arch_str = self._determine_target_arch_str(target)
        arch_path = os.path.join(database_path, arch_str)
        if not os.path.exists(arch_path):
            logger.info(
                f"Target {arch_str} does not exist in the database, skipping loading operators from the database"
            )
            return
        self._load_operators_from_arch_path(arch_path, target)

    def _ensure_database_path(self, database_path):
        if database_path is None:
            return tempfile.mkdtemp()
        os.makedirs(database_path, exist_ok=True)
        return database_path

    def _determine_arch_str(self, op_inst, target):
        return (target if target else "-".join(list(op_inst.target.keys) + [op_inst.target.arch]))

    def _ensure_directory(self, path):
        os.makedirs(path, exist_ok=True)

    def _save_operator_config_and_artifact(self, config, op_inst, config_path):
        config_type, operator_type = type(config).__name__, type(op_inst).__name__
        with open(os.path.join(config_path, f"{config_type}.json"), "w") as json_file:
            json.dump(asdict(config), json_file)
        artifact_path = os.path.join(config_path, "tvm_rt_mod." + tar.output_format)
        try:
            op_inst.rt_mod.export_library(artifact_path, fcompile=tar)
        except Exception as e:
            # library does not support export_library
            export_error = e  # noqa: F841
            pass
        json_data = {"config_type": config_type, "operator_type": operator_type}
        json_file_path = os.path.join(config_path, "mapping.json")
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file)

        # For writing source.cu file
        source_file_path = os.path.join(config_path, "source.cu")
        with open(source_file_path, "w") as source_file:
            source_file.write(op_inst.get_source())

        # For writing optimized.py file
        optimized_file_path = os.path.join(config_path, "optimized.py")
        with open(optimized_file_path, "w") as optimized_file:
            if op_inst.optimized_func is not None:
                optimized_file.write(op_inst.optimized_func.script(show_meta=False))
        if op_inst.wrapper.lib_name is not None:
            # copy lib name to the same directory as the artifact
            src_name = op_inst.wrapper.src_name
            shutil.copy(
                src_name,
                os.path.join(config_path, os.path.basename("wrapper_source.cu")),
            )
            lib_name = op_inst.wrapper.lib_name
            shutil.copy(
                lib_name,
                os.path.join(config_path, os.path.basename("wrapper_compiled.so")),
            )

    def _determine_target_arch_str(self, target):
        return (target if isinstance(target, str) else "-".join(list(target.keys) + [target.arch]))

    def _load_operators_from_arch_path(self, arch_path, target):
        for root, dirs, _ in os.walk(arch_path):
            for directory in dirs:
                config_path = os.path.join(root, directory)
                self._load_operator(config_path, target)

    def _load_operator(self, config_path, target):
        mapping, config, rt_mod, src_name, lib_name = None, None, None, None, None
        for file in os.listdir(config_path):
            full_path = os.path.join(config_path, file)
            if file == "mapping.json":
                with open(full_path) as f:
                    mapping = json.load(f)
            elif file.endswith(".json"):
                with open(full_path) as f:
                    config = json.load(f)
            elif file.endswith(".tar"):
                rt_mod = tvm.runtime.load_module(full_path)
            elif file == "wrapper_compiled.so":
                lib_name = full_path
            elif file == "wrapper_source.cu":
                src_name = full_path

        if mapping and config and rt_mod:
            self._instantiate_and_add_operator(mapping, config, rt_mod, src_name, lib_name, target)

    def _instantiate_and_add_operator(self, mapping, config, rt_mod, src_name, lib_name, target):
        config_cls = getattr(bitblas, mapping["config_type"])
        operator_cls = getattr(bitblas, mapping["operator_type"])
        op_inst = operator_cls(
            config=config_cls(**config), target=target, enable_tuning=False, from_database=True)
        op_inst.update_runtime_module(rt_mod, src_name=src_name, lib_name=lib_name)
        self.add(config_cls(**config), op_inst)


global_operator_cache = OperatorCache()


def load_global_ops_cache(database_path=BITBLAS_DATABASE_PATH, target=None):
    if target is None:
        target = bitblas.auto_detect_nvidia_target()
    logger.info(f"Loading operators from database {database_path} for target {target}")
    global_operator_cache.load_from_database(database_path, target)
    return global_operator_cache


def get_database_path():
    return BITBLAS_DATABASE_PATH


def set_database_path(path):
    global BITBLAS_DATABASE_PATH
    BITBLAS_DATABASE_PATH = path
    return BITBLAS_DATABASE_PATH

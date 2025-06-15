# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Callable, Dict

import torch

import vllm.envs as envs

logger = logging.getLogger(__name__)

# make sure one process only loads plugins once
plugins_loaded = False

# todo 通过group加载插件
def load_plugins_by_group(group: str) -> Dict[str, Callable]:
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    allowed_plugins = envs.VLLM_PLUGINS
    # todo 发现插件，比如
    #  entry_points={
    #         'vllm.platform_plugins': ["kunlun = vllm_xpu:register"],
    #         'vllm.general_plugins': ["kunlun_model = vllm_xpu:register_model"],
    #  })
    # todo 那么entry_points(group=vllm.general_plugins)，
    #  应该返回[EntryPoint(name='kunlun_model', value='vllm_xpu:register_model', group='vllm.general_plugins')]
    discovered_plugins = entry_points(group=group)
    if len(discovered_plugins) == 0:
        logger.debug("No plugins for group %s found.", group)
        return {}
    logger.info("Available plugins for group %s:", group)
    for plugin in discovered_plugins:
        logger.info("name=%s, value=%s", plugin.name, plugin.value)
    if allowed_plugins is None:
        logger.info("all available plugins for group %s will be loaded.",
                    group)
        logger.info("set environment variable VLLM_PLUGINS to control"
                    " which plugins to load.")
    plugins = {}
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            try:
                # todo plugin.load() 会动态导入vllm_xpu模块，并返回register_model函数
                # todo 具体做法就是 1、通过正则，将value='vllm_xpu:register_model'，转化为{'module': 'vllm_xpu', 'attr': 'register_model', 'extras': None}
                # todo            2、导入model：vllm_xpu，然后在vllm_xpu中寻找函数register_model
                func = plugin.load()
                plugins[plugin.name] = func
                logger.info("plugin %s loaded.", plugin.name)
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)
    # todo 返回{'kunlun_model': <function register_model at 0x7f8c1a2b3e20>}
    return plugins


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global plugins_loaded
    if plugins_loaded:
        return
    plugins_loaded = True

    # some platform-specific configurations
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        # see https://github.com/pytorch/pytorch/blob/43c5f59/torch/_dynamo/config.py#L158
        torch._dynamo.config.disable = True
    elif current_platform.is_hpu():
        # NOTE(kzawora): PT HPU lazy backend (PT_HPU_LAZY_MODE = 1)
        # does not support torch.compile
        # Eager backend (PT_HPU_LAZY_MODE = 0) must be selected for
        # torch.compile support
        is_lazy = os.environ.get('PT_HPU_LAZY_MODE', '1') == '1'
        if is_lazy:
            torch._dynamo.config.disable = True
            # NOTE(kzawora) multi-HPU inference with HPUGraphs (lazy-only)
            # requires enabling lazy collectives
            # see https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html # noqa: E501
            os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = 'true'
    # todo 按照vllm.general_plugins加载插件，返回{'kunlun_model': <function register_model at 0x7f8c1a2b3e20>}
    plugins = load_plugins_by_group(group='vllm.general_plugins')
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():
        # todo 执行register_model函数，将自定义的模型注册到vllm中
        func()

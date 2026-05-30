"""
装饰器工具模块

提供重试装饰器等通用装饰器
"""

import time
from typing import Callable


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    重试装饰器
    
    当函数执行失败时自动重试指定次数
    
    Args:
        max_retries: 最大重试次数，默认3次
        delay: 重试间隔时间（秒），默认1秒
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
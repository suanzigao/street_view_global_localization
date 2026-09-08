"""
工具模块 - 为Agent提供各种工具能力
"""

from .sam3_tool import SAM3Tool, TOOL_DESCRIPTION, tool_call_segment

__all__ = ['SAM3Tool', 'TOOL_DESCRIPTION', 'tool_call_segment']

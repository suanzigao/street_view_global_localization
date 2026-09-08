"""
SAM3 工具模块 - 为 Agent 提供视觉分割能力

功能：
1. 支持多类别掩码叠加到同一图像（统一颜色）
2. 自动保存分割结果图像到本地路径
3. 保持图像分辨率不变，掩码边缘清晰
"""

import os
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

try:
    from sam3api import SAM3Segmenter
except ImportError:
    SAM3Segmenter = None


class SAM3Tool:
    """SAM3 分割工具封装"""

    DEFAULT_MASK_COLOR = (255, 0, 0, 180)  # 单一颜色：半透明红色

    def __init__(self):
        self.segmenter = None
        self.available = SAM3Segmenter is not None

    def initialize(self):
        if self.available and self.segmenter is None:
            self.segmenter = SAM3Segmenter()
            self.segmenter.initialize()
        return self

    def _ensure_valid_image(self, image: Image.Image) -> Image.Image:
        """确保图像格式是SAM3支持的RGB格式"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def _ensure_valid_mask(self, mask):
        """确保掩码格式正确"""
        mask_np = np.array(mask)

        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]

        if mask_np.ndim == 1:
            size = int(np.sqrt(len(mask_np)))
            if size * size == len(mask_np):
                mask_np = mask_np.reshape((size, size))
            else:
                return None

        if mask_np.ndim == 2:
            return mask_np

        return None

    def _resize_mask_to_image(self, mask: np.ndarray, target_size: tuple) -> np.ndarray:
        """将掩码调整为与目标图像相同的尺寸"""
        if mask.shape[:2] == target_size[:2][::-1]:
            return mask
        mask_img = Image.fromarray(mask.astype(np.uint8), mode='L')
        resized = mask_img.resize(target_size[:2][::-1], Image.NEAREST)
        return np.array(resized)

    def segment(self, image: Image.Image, prompt: str):
        """
        执行图像分割

        Args:
            image: PIL图像对象
            prompt: 文本提示词

        Returns:
            dict: 包含 masks, bboxes, scores 的结果
        """
        if not self.available:
            return {'success': False, 'error': 'SAM3 not available'}

        try:
            valid_image = self._ensure_valid_image(image)

            self.segmenter.set_image(image=valid_image)
            self.segmenter.segment(prompt)

            masks = self.segmenter.get_masks()
            bboxes = self.segmenter.get_bboxes()
            scores = self.segmenter.get_scores()

            valid_masks = []
            for mask in masks:
                valid_mask = self._ensure_valid_mask(mask)
                if valid_mask is not None:
                    resized_mask = self._resize_mask_to_image(valid_mask, valid_image.size)
                    valid_masks.append(resized_mask)

            return {
                'success': True,
                'prompt': prompt,
                'masks': valid_masks,
                'bboxes': bboxes,
                'scores': scores,
                'count': len(valid_masks)
            }
        except Exception as e:
            import traceback
            return {'success': False, 'error': f"{str(e)}\n{traceback.format_exc()}"}

    def overlay_single_mask(self, image: Image.Image, mask: np.ndarray, color: tuple = None) -> Image.Image:
        """
        将单个掩码叠加到图像上（保持清晰边缘）

        Args:
            image: 原始图像（PIL Image）
            mask: 掩码数组（numpy array）
            color: 叠加颜色（RGBA）

        Returns:
            Image: 带单个掩码的图像
        """
        if color is None:
            color = self.DEFAULT_MASK_COLOR

        try:
            image_rgba = image.convert('RGBA')

            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)

            # 将掩码值乘以alpha比例，确保半透明效果
            alpha_mask = (mask * (color[3] / 255)).astype(np.uint8)
            mask_img = Image.fromarray(alpha_mask, mode='L')

            colored_mask = Image.new('RGBA', image.size, color[:3] + (255,))
            colored_mask.putalpha(mask_img)

            result = Image.alpha_composite(image_rgba, colored_mask)
            return result

        except Exception as e:
            return image.convert('RGBA')

    def overlay_masks_combined(
        self,
        image: Image.Image,
        segmentation_results: list,
        color: tuple = None,
        alpha: int = 180
    ) -> Image.Image:
        """
        将多个类别的掩码叠加到同一图像上（统一颜色）

        Args:
            image: 原始图像（PIL Image）
            segmentation_results: 分割结果列表，每个元素包含 {'prompt': str, 'masks': [np.array], 'bboxes': [], 'scores': []}
            color: 掩码颜色（RGBA）
            alpha: 掩码透明度 (0-255)

        Returns:
            Image: 叠加了所有掩码的RGBA图像
        """
        if color is None:
            color = self.DEFAULT_MASK_COLOR

        # 确保颜色带有 alpha 通道
        if len(color) == 3:
            color = (*color, alpha)
        else:
            color = (*color[:3], alpha)

        image_rgba = image.convert('RGBA')
        result_image = image_rgba.copy()

        for idx, seg_result in enumerate(segmentation_results):
            if not seg_result.get('success') or not seg_result.get('masks'):
                continue

            masks = seg_result.get('masks', [])

            for mask in masks:
                try:
                    resized_mask = self._resize_mask_to_image(mask, image.size)
                    
                    # 将掩码值乘以alpha比例，确保半透明效果
                    # mask值范围是0-255，乘以alpha/255后得到带透明度的掩码
                    alpha_mask = (resized_mask * (color[3] / 255)).astype(np.uint8)
                    mask_img = Image.fromarray(alpha_mask, mode='L')

                    colored_mask = Image.new('RGBA', image.size, color[:3] + (255,))
                    colored_mask.putalpha(mask_img)

                    result_image = Image.alpha_composite(result_image, colored_mask)

                except Exception as e:
                    continue

        return result_image

    def get_segmented_images(
        self,
        image: Image.Image,
        prompts: list,
        save_path: str = None,
        output_format: str = 'PNG'
    ):
        """
        对多个提示词进行分割，并返回带叠加掩码的图像

        Args:
            image: 原始图像（PIL Image）
            prompts: 提示词列表，如 ["building", "road sign", "car"]
            save_path: 保存路径（可选），如 "output/segmented.png"
            output_format: 输出格式，'PNG' 或 'JPG'

        Returns:
            dict: 包含以下键值的字典
                - 'success': bool, 是否成功
                - 'combined_image_b64': str, 所有掩码叠加后的图像（Base64）
                - 'combined_image_path': str, 保存路径（如果指定了save_path）
                - 'results': list, 每个类别的分割结果
                - 'prompt_count': int, 成功分割的类别数
        """
        if not self.available:
            return {
                'success': False,
                'error': 'SAM3 not available',
                'combined_image_b64': None,
                'combined_image_path': None,
                'results': [],
                'prompt_count': 0
            }

        try:
            valid_image = self._ensure_valid_image(image)

            segmentation_results = []
            for prompt in prompts:
                result = self.segment(valid_image, prompt)
                segmentation_results.append(result)

            success_count = sum(1 for r in segmentation_results if r.get('success') and r.get('count', 0) > 0)

            combined_image = self.overlay_masks_combined(
                valid_image,
                segmentation_results,
                alpha=180
            )

            buffered = BytesIO()
            if output_format.upper() == 'JPG' or output_format.upper() == 'JPEG':
                rgb_image = combined_image.convert('RGB')
                rgb_image.save(buffered, format='JPEG', quality=95)
            else:
                combined_image.save(buffered, format='PNG')
            combined_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            saved_path = None
            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                if output_format.upper() == 'JPG' or output_format.upper() == 'JPEG':
                    rgb_image = combined_image.convert('RGB')
                    rgb_image.save(save_path, format='JPEG', quality=95)
                else:
                    combined_image.save(save_path, format='PNG')
                saved_path = save_path

            results_detail = []
            for idx, result in enumerate(segmentation_results):
                results_detail.append({
                    'prompt': result.get('prompt', prompts[idx]),
                    'success': result.get('success', False),
                    'count': result.get('count', 0),
                    'masks': result.get('masks', []),
                    'bboxes': result.get('bboxes', []),
                    'scores': result.get('scores', [])
                })

            return {
                'success': True,
                'combined_image_b64': combined_b64,
                'combined_image_path': saved_path,
                'results': results_detail,
                'prompt_count': success_count,
                'image_size': valid_image.size,
                'output_format': output_format
            }

        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': f"{str(e)}\n{traceback.format_exc()}",
                'combined_image_b64': None,
                'combined_image_path': None,
                'results': [],
                'prompt_count': 0
            }

    def save_segmented_image(
        self,
        image: Image.Image,
        segmentation_results: list,
        save_path: str,
        color: tuple = None,
        output_format: str = 'PNG'
    ):
        """
        保存分割结果图像到指定路径

        Args:
            image: 原始图像
            segmentation_results: 分割结果列表
            save_path: 保存路径
            color: 掩码颜色
            output_format: 输出格式

        Returns:
            bool: 是否保存成功
        """
        try:
            combined_image = self.overlay_masks_combined(image, segmentation_results, color)

            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

            if output_format.upper() == 'JPG' or output_format.upper() == 'JPEG':
                rgb_image = combined_image.convert('RGB')
                rgb_image.save(save_path, format='JPEG', quality=95)
            else:
                combined_image.save(save_path, format='PNG')

            return True

        except Exception as e:
            print(f"保存分割图像失败: {e}")
            return False

    def close(self):
        if self.segmenter:
            self.segmenter.close()


def tool_call_segment(image_b64: str, prompt: str):
    """
    工具调用接口 - 从Base64图像和提示词进行分割

    Args:
        image_b64: Base64编码的图像
        prompt: 文本提示词

    Returns:
        dict: 分割结果
    """
    tool = SAM3Tool()
    try:
        tool.initialize()

        img_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(img_data))

        result = tool.segment(image, prompt)
        return result
    finally:
        tool.close()


TOOL_DESCRIPTION = {
    "name": "sam3_segmentation",
    "description": "使用SAM3模型对图像进行文本引导分割，识别图像中的特定物体。可用于识别建筑、车辆、路牌、地标等视觉特征，辅助地理定位。支持多类别分割掩码叠加到同一图像，并自动保存结果。",
    "parameters": {
        "image_b64": {
            "type": "string",
            "description": "待分割图像的Base64编码"
        },
        "prompts": {
            "type": "array",
            "description": "要识别的物体描述列表，如 [\"building\", \"car\", \"road sign\"]"
        },
        "save_path": {
            "type": "string",
            "description": "可选，保存路径，如 \"output/segmented.png\""
        },
        "output_format": {
            "type": "string",
            "description": "输出格式，'PNG' 或 'JPG'，默认 'PNG'"
        }
    },
    "returns": {
        "type": "object",
        "description": "包含分割结果的字典：combined_image_b64（叠加掩码图像的Base64）, combined_image_path（保存路径）, prompt_count（成功分割的类别数）"
    }
}


if __name__ == "__main__":
    tool = SAM3Tool()
    if not tool.available:
        print("SAM3 不可用，请检查依赖安装")
    else:
        try:
            test_image_path = "models/sam3/assets/images/test_image.jpg"
            if os.path.exists(test_image_path):
                test_image = Image.open(test_image_path)

                prompts = ["building", "car", "road sign"]

                result = tool.get_segmented_images(
                    test_image,
                    prompts,
                    save_path="test_segmented_combined.png",
                    output_format='PNG'
                )

                if result['success']:
                    print(f"✅ 分割成功，成功分割 {result['prompt_count']} 个类别")
                    print(f"📷 叠加掩码图像已保存到: {result['combined_image_path']}")
                    print(f"📐 图像尺寸: {result['image_size']}")
                    print(f"📦 Base64长度: {len(result['combined_image_b64'])} 字符")
                else:
                    print(f"❌ 分割失败: {result.get('error')}")
            else:
                print(f"测试图像不存在: {test_image_path}")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            tool.close()
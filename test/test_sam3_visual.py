import sys
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, '/home/suanzigao/PycharmProjects/street_view_global_localization')
from sam3api import SAM3Segmenter, segment_image


def overlay_mask_on_image(original_image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    将分割掩码叠加到原图上

    Args:
        original_image: PIL Image
        mask: numpy array (H, W) 或 (1, H, W)
        alpha: 掩码透明度
        color: 掩码颜色 (R, G, B)

    Returns:
        PIL Image
    """
    # 确保掩码形状正确
    if len(mask.shape) == 3:
        mask = mask[0]

    # 将掩码转为布尔值
    mask_bool = mask > 127

    # 创建掩码图像
    mask_img = Image.new('RGBA', original_image.size, (*color, 0))
    mask_array = np.array(mask_img)
    mask_array[mask_bool, :3] = color
    mask_array[mask_bool, 3] = int(alpha * 255)
    mask_img = Image.fromarray(mask_array)

    # 叠加到原图
    result = Image.alpha_composite(original_image.convert('RGBA'), mask_img)
    return result.convert('RGB')


def draw_bbox_on_image(image, bbox, color=(0, 255, 0), width=3):
    """
    在图像上绘制边界框

    Args:
        image: PIL Image
        bbox: [x1, y1, x2, y2]
        color: 颜色 (R, G, B)
        width: 线条宽度

    Returns:
        PIL Image
    """
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return image


def draw_text_on_image(image, text, position, color=(255, 255, 255), font_size=20):
    """
    在图像上绘制文本

    Args:
        image: PIL Image
        text: 要绘制的文本
        position: (x, y)
        color: 颜色

    Returns:
        PIL Image
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 绘制文本背景
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 180))
    draw.text(position, text, fill=color, font=font)
    return image


def visualize_single_segmentation(image_path, prompt, output_path, return_all=False):
    """
    可视化单目标分割结果

    Args:
        image_path: 输入图像路径
        prompt: 分割提示词
        output_path: 输出图像路径
        return_all: 是否可视化所有结果
    """
    print(f"正在处理: {image_path}")
    print(f"提示词: {prompt}")

    # 分割
    result = segment_image(image_path, prompt, return_all=return_all)

    if not result['success']:
        print(f"分割失败: {result['error']}")
        return

    # 加载原图
    original_img = Image.open(image_path)
    print(f"找到 {result['count']} 个分割结果")

    if result['count'] == 0:
        print("未找到任何目标")
        original_img.save(output_path)
        return

    if return_all and 'masks' in result:
        # 可视化所有结果
        current_img = original_img.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        for i, (mask, bbox, score) in enumerate(zip(result['masks'], result['bboxes'], result['scores'])):
            color = colors[i % len(colors)]
            current_img = overlay_mask_on_image(current_img, mask, alpha=0.4, color=color)
            current_img = draw_bbox_on_image(current_img, bbox, color=color, width=3)

        # 添加说明文本
        current_img = draw_text_on_image(
            current_img,
            f"Prompt: {prompt}, Found: {result['count']}",
            (10, 10),
            font_size=24
        )
    else:
        # 只可视化第一个结果
        current_img = overlay_mask_on_image(original_img, result['mask'], alpha=0.5, color=(255, 0, 0))
        current_img = draw_bbox_on_image(current_img, result['bbox'], color=(0, 255, 0), width=3)
        score_text = f"Score: {result['score']:.3f}"
        current_img = draw_text_on_image(
            current_img,
            f"Prompt: {prompt}, {score_text}",
            (10, 10),
            font_size=24
        )

    # 保存结果
    current_img.save(output_path)
    print(f"结果已保存至: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SAM-3 API 可视化测试')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--prompt', type=str, required=True, help='分割提示词')
    parser.add_argument('--output', type=str, default=None, help='输出图像路径')
    parser.add_argument('--all', action='store_true', help='可视化所有分割结果')

    args = parser.parse_args()

    # 确定输出路径
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_segmented{ext}"

    # 执行可视化
    visualize_single_segmentation(args.image, args.prompt, args.output, return_all=args.all)


if __name__ == "__main__":
    main()

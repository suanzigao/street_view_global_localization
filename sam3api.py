import torch
from PIL import Image
import numpy as np

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Segmenter:
    def __init__(self):
        self.model = None
        self.processor = None
        self.inference_state = None
        self._is_initialized = False

    def initialize(self):
        if not self._is_initialized:
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            self._is_initialized = True
        return self

    def set_image(self, image_path=None, image=None):
        if not self._is_initialized:
            self.initialize()

        if image_path is not None:
            image = Image.open(image_path)
        elif image is None:
            raise ValueError("Either image_path or image must be provided")

        self.inference_state = self.processor.set_image(image)
        return self

    def segment(self, prompt):
        if self.inference_state is None:
            raise ValueError("Image not set. Call set_image() first.")

        self.inference_state = self.processor.set_text_prompt(
            prompt=prompt,
            state=self.inference_state
        )
        return self.inference_state

    def get_masks(self):
        if self.inference_state is None:
            raise ValueError("No inference state available. Call segment() first.")

        masks = self.inference_state.get("masks")
        if masks is None:
            return []

        mask_list = []
        for mask in masks:
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            mask_list.append(mask_np.astype(np.uint8) * 255)
        return mask_list

    def get_mask(self, index=0):
        masks = self.get_masks()
        if index >= len(masks):
            raise IndexError(f"Mask index {index} out of range, only {len(masks)} masks available")
        return masks[index]

    def get_bboxes(self):
        if self.inference_state is None:
            raise ValueError("No inference state available. Call segment() first.")

        boxes = self.inference_state.get("boxes")
        if boxes is None:
            return []

        bbox_list = []
        for box in boxes:
            box_np = box.cpu().numpy() if hasattr(box, 'cpu') else box
            bbox_list.append(box_np.tolist())
        return bbox_list

    def get_bbox(self, index=0):
        bboxes = self.get_bboxes()
        if index >= len(bboxes):
            raise IndexError(f"Bbox index {index} out of range, only {len(bboxes)} bboxes available")
        return bboxes[index]

    def get_scores(self):
        if self.inference_state is None:
            raise ValueError("No inference state available. Call segment() first.")

        scores = self.inference_state.get("scores")
        if scores is None:
            return []

        score_list = []
        for score in scores:
            score_val = score.cpu().item() if hasattr(score, 'cpu') else score
            score_list.append(score_val)
        return score_list

    def get_segmentation_count(self):
        if self.inference_state is None:
            return 0
        masks = self.inference_state.get("masks")
        if masks is None:
            return 0
        return len(masks)

    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        self.clear_cache()
        self.model = None
        self.processor = None
        self.inference_state = None
        self._is_initialized = False


def segment_image(image_path, prompt, return_all=False):
    segmenter = SAM3Segmenter()
    try:
        segmenter.set_image(image_path)
        segmenter.segment(prompt)

        mask_count = segmenter.get_segmentation_count()
        if mask_count == 0:
            return {
                'masks': [],
                'bboxes': [],
                'scores': [],
                'success': True,
                'count': 0
            }

        if return_all:
            return {
                'masks': segmenter.get_masks(),
                'bboxes': segmenter.get_bboxes(),
                'scores': segmenter.get_scores(),
                'success': True,
                'count': mask_count
            }
        else:
            return {
                'mask': segmenter.get_mask(0),
                'bbox': segmenter.get_bbox(0),
                'score': segmenter.get_scores()[0] if segmenter.get_scores() else None,
                'success': True,
                'count': mask_count
            }
    except Exception as e:
        return {
            'mask': None,
            'masks': [],
            'bboxes': [],
            'scores': [],
            'success': False,
            'error': str(e)
        }
    finally:
        segmenter.close()


if __name__ == "__main__":
    result = segment_image("models/sam3/assets/images/test_image.jpg", "shoe")
    if result['success']:
        print(f"分割成功，找到 {result['count']} 个目标")
        if result['count'] > 0:
            print(f"Mask 形状: {result['mask'].shape}")
            print(f"Bounding box: {result['bbox']}")
    else:
        print(f"分割失败: {result['error']}")

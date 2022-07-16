# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utils for RetinaFace inference."""

import numpy as np
from PIL import ImageFont, ImageDraw, Image


def ndarray2image(ndarray):
    """Convert numpy.ndarray image to PIL image."""
    ndarray = ndarray[:, :, ::-1]
    image = Image.fromarray(ndarray)
    return image


def image2ndarray(image):
    """Convert PIL image to numpy.ndarray image."""
    ndarray = np.array(image)
    return ndarray[:, :, ::-1]


def draw_confidence(conf, top, left, draw, font):
    """
    Draw confidence above bounding boxes for images.

    Args:
        conf (float): The confidence number to draw.
        top (int): Top coordinate of bounding box.
        left (int): Left coordinate of bounding box.
        draw (PIL.ImageDraw): A drawing instance from PIL.
        font (numpy.ndarray): A font object, usually created by PIL.ImageFont.truetype().
    """
    conf_text = '%.6f' % conf
    label_size = draw.textsize(conf_text, font)
    conf_text = conf_text.encode('utf-8')
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill='red')
    draw.text(text_origin, str(conf_text, 'UTF-8'), fill=(0, 0, 0), font=font)


def draw_preds(frame, bbox_list, draw_conf=False, landmark_list=None):
    """
    Draw predict boxes and landmarks for image.

    Args:
        frame (numpy.ndarray): Frame of images, usually get from cv2.imread, a [H,W,C] shape tensor.
        bbox_list (list): A list of lists, each list in bbox_list is [x, y, w, h, conf] represents a prediction box.
        draw_conf (bool): Whether draw confidence number above boxes.
        landmark_list (numpy.ndarray): Shape [N,10], represents 5 landmark x,y pairs of N faces.

    Returns:
        Numpy ndarray, represents the image with boxes and landmarks on it.
    """
    image = ndarray2image(frame)
    thickness = int(
        max((image.size[0] + image.size[1]) // np.mean(np.array(image.size[:2])), 1))
    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    for i, j in enumerate(bbox_list):
        x, y, width, height, conf = j
        left, top, right, bottom = x, y, x + width, y + height
        top = max(0, int(top))
        left = max(0, int(left))
        bottom = min(image.size[1], int(bottom))
        right = min(image.size[0], int(right))
        draw = ImageDraw.Draw(image)
        for k in range(thickness):
            draw.rectangle([left + k, top + k, right - k, bottom - k], outline='red')
        if landmark_list:
            for k in range(5):
                center_x, center_y = (landmark_list[i][k * 2], landmark_list[i][k * 2 + 1])
                mult_thick = thickness * 0.2
                draw.ellipse(
                    (center_x - mult_thick, center_y - mult_thick, center_x + mult_thick, center_y + mult_thick),
                    fill='blue')
        if draw_conf:
            draw_confidence(conf, top, left, draw, font)
    return image2ndarray(image)


def contain_chinese(string):
    """Check whether the string contains Chinese characters."""
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def cast_list_to_int(input_list):
    """Converts the elements of a list in a list to int."""
    processed_list = []
    for i in input_list:
        cur_list = []
        for j in i:
            cur_list.append(int(j))
        processed_list.append(cur_list)
    return processed_list

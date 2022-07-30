# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections
from importlib import reload
import matplotlib.pyplot as plt

reload(sys)
import cv2

KR_CHARSET = None

def get_offset(ch, font, canvas_size):
    font_size = font.getsize(ch)
    font_offset = font.getoffset(ch)
    offset_x = canvas_size/2 - font_size[0]/2 - font_offset[0]/2
    offset_y = canvas_size/2 - font_size[1]/2 - font_offset[1]/2
    return [ offset_x, offset_y ]


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255)).convert('L')
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


def draw_example(ch, src_font, dst_font, canvas_size, src_offset, dst_offset, filter_hashes):
    dst_img = draw_single_char(ch, dst_font, canvas_size, dst_offset[0], dst_offset[1])
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
    src_img = draw_single_char(ch, src_font, canvas_size, src_offset[0], src_offset[1])
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def get_font_offset(charset, font, canvas_size, filter_hashes):
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    font_offset = np.array([0, 0])
    count = 0
    for c in sample:
        font_img = draw_single_char(c, font, canvas_size, 0, 0)
        font_hash = hash(font_img.tobytes())
        if not font_hash in filter_hashes:
            np.add(font_offset, get_offset(c, font, canvas_size), out=font_offset, casting="unsafe")
            count += 1
    np.divide(font_offset,count, out=font_offset,casting="unsafe")
    return font_offset

def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]

def select_sample(charset):
    # this returns 399 samples from KR charset
    # we selected 399 characters to sample as uniformly as possible
    # (the number of each ChoSeong is fixed to 21 (i.e., 21 Giyeok, 21 Nieun ...))
    # Given the designs of these 399 characters, the rest of Hangeul will be generated
    samples = []
    for i in range(399):
        samples.append(charset[28*i+(i%28)])
    np.random.shuffle(samples)
    return samples


def draw_handwriting(ch, src_font, canvas_size, src_offset, dst_folder):
    #s = ch.decode('utf-8').encode('raw_unicode_escape').replace("\\u","").upper()
    s = ch.encode('raw_unicode_escape').decode('utf-8').replace("\\u","").upper()
    dst_path = dst_folder + "/uni" + s + ".png"
    if not os.path.exists(dst_path):
        return
    #dst_img = Image.open(dst_path)

    #글자 센터 맞추기
    dst_img = cv2.imread(dst_path)
    img_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
    ret, img_th = cv2.threshold(img_gray, 100, 230, cv2.THRESH_BINARY_INV)
    contours, hierachy= cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(each) for each in contours]

    coord = [150,150,0,0]
    for i in range(len(rects)): #글자 bounding box 좌표 찾기
        coord[0] = min(rects[i][0],coord[0])
        coord[1] = min(rects[i][1],coord[1])
        coord[2] = max(rects[i][0]+rects[i][2], coord[2])
        coord[3] = max(rects[i][1]+rects[i][3], coord[3])

    x = np.median([coord[0],coord[2]])
    y = np.median([coord[1],coord[3]])
    dx = int(64-x)
    dy = int(64-y)
    mtrx = np.float32([[1,0,dx], [0,1,dy]])
    rows,cols = dst_img.shape[0:2] 
    dst = cv2.warpAffine(dst_img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, 1, (255,0,0) ) #가운데 정렬
    dst_img = dst[:128,:128]

    #erosion
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    output = cv2.erode(dst_img,k)
    dst_img = Image.fromarray(output)
    
    # check the filter example in the hashes or not
    src_img = draw_single_char(ch, src_font, canvas_size, src_offset[0], src_offset[1])
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img

def font2img(src, dst, charset, char_size, canvas_size,
             x_offset, y_offset, sample_count, sample_dir, label=0, filter_by_hash=True, fixed_sample=False, all_sample=False, handwriting_dir=False):
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    dst_filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, 0, 0))
    dst_offset = get_font_offset(charset, dst_font, canvas_size, dst_filter_hashes)
    print("Src font offset : ", [x_offset, y_offset])
    print("Dst font offset : ", dst_offset)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, dst_offset[0], dst_offset[1]))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    if handwriting_dir:
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        train_set = []
        for c in charset:
            e = draw_handwriting(c, src_font, canvas_size, [x_offset, y_offset], handwriting_dir)
            if e:
                e.save(os.path.join(sample_dir, "%d_%s_train.png" % (label, c.replace("\\u","").upper())))
                train_set.append(c)
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)

        #np.random.shuffle(charset)
        #count = 0
        #for c in charset:
        #    e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes=set())
        #    if e:
        #        e.save(os.path.join(sample_dir, "%d_%s_val.png" % (label, c.replace("\\u","").upper())))
        #        count += 1
        #        if count % 100 == 0:
        #            print("processed %d chars" % count)
        return

    if fixed_sample:
        train_set = select_sample(charset)
        for c in train_set:
            e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes)
            if e:
                e.save(os.path.join(sample_dir, "%d_%04d_train.png" % (label, count)))
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)

        np.random.shuffle(charset)
        count = 0
        for c in charset:
            if count == sample_count:
                break
            if c in train_set:
                continue
            e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes=set())
            if e:
                e.save(os.path.join(sample_dir, "%d_%04d_val.png" % (label, count)))
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)
        return

    if all_sample:
        for c in charset:
            e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes)
            if e:
                e.save(os.path.join(sample_dir, "%d_%04d.png" % (label, count)))
                count += 1
                if count % 1000 == 0:
                    print("processed %d chars" % count)
        return

    for c in charset:
        if count == sample_count:
            break
        e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.png" % (label, count)))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)

import easydict
args = easydict.EasyDict({
    "src_font": "/content/drive/MyDrive/neural-fonts-master/newfonts2/NanumGothic.ttf",
    "dst_font": "/content/drive/MyDrive/neural-fonts-master/newfonts2/NanumGothic.ttf",
    "filter" : 0,
    "shuffle" : 0,
    "char_size" : 80,
    "canvas_size" : 128,
    "x_offset" : 27,
    "y_offset" : 16,
    "sample_count" : 1000,
    "sample_dir" : "/content/drive/MyDrive/neural-fonts-master/newfonts2/inference3",
    "label" : 0,
    "fixed_sample" : 0,
    "all_sample" : 0,
    "handwriting_dir" : "/content/drive/MyDrive/neural-fonts-master/newfonts2/inferencedata"
})

if __name__ == "__main__":
    charset = []
    for i in range(0xac00,0xd7a4):
        charset.append(chr(i))
    for i in range(0x3131,0x3164):
        charset.append(chr(i))
    if args.shuffle:
        np.random.shuffle(charset)
    font2img(args.src_font, args.dst_font, charset, args.char_size,
             args.canvas_size, args.x_offset, args.y_offset,
             args.sample_count, args.sample_dir, args.label, args.filter, args.fixed_sample, args.all_sample, args.handwriting_dir)

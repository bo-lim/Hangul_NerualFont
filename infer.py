# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import argparse
from model.unet import UNet
from model.utils import compile_frames_to_gif

import easydict

"""
People are made to have fun and be 中二 sometimes
                                --Bored Yan LeCun
"""

#parser = argparse.ArgumentParser(description='Inference for unseen data')
#parser.add_argument('--model_dir', dest='model_dir', required=True,
#                    help='directory that saves the model checkpoints')
#parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
#parser.add_argument('--source_obj', dest='source_obj', type=str, required=True, help='the source images for inference')
#parser.add_argument('--embedding_ids', default='embedding_ids', type=str, help='embeddings involved')
#parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
#parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
#                    help='use conditional instance normalization in your model')
#parser.add_argument('--interpolate', dest='interpolate', type=int, default=0,
#                    help='interpolate between different embedding vectors')
#parser.add_argument('--steps', dest='steps', type=int, default=10, help='interpolation steps in between vectors')
#parser.add_argument('--output_gif', dest='output_gif', type=str, default=None, help='output name transition gif')
#parser.add_argument('--uroboros', dest='uroboros', type=int, default=0,
#                    help='Shōnen yo, you have stepped into uncharted territory')
#parser.add_argument('--compare', dest='compare', type=int, default=0, help='Compare with original font image')
#parser.add_argument('--show_ssim', dest='show_ssim', type=int, default=0, help='Visualize ssim in the result image')
#parser.add_argument('--progress_file', dest='progress_file', type=str, default=None, help='Progress file name. Not used with compare')
#args = parser.parse_args()

args = easydict.EasyDict({
    "model_dir": "/content/drive/MyDrive/neural-fonts-master/newfonts/checkpoint/experiment_0_batch_16",
    "batch_size": 1,
    "source_obj" : "/content/drive/MyDrive/neural-fonts-master/newfonts/data/val.obj",
    "embedding_ids" : "0",
    "save_dir" : "/content/drive/MyDrive/neural-fonts-master/newfonts/save",
    "inst_norm" : 0,
    "interpolate" : 0,
    "steps" : 10,
    "y_offset" : 16,
    "output_gif" : None,
    "uroboros" : 0,
    "compare" : 0,
    "show_ssim" : 0,
    "progress_file" : "/content/drive/MyDrive/neural-fonts-master/newfonts/logs/progress"
})

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=args.batch_size)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=args.inst_norm)
        embedding_ids = [int(i) for i in args.embedding_ids.split(",")]
        if not args.interpolate:
            if len(embedding_ids) == 1:
                embedding_ids = embedding_ids[0]
            if args.compare:
                model.infer_compare(model_dir=args.model_dir, source_obj=args.source_obj, embedding_ids=embedding_ids,
                        save_dir=args.save_dir, show_ssim=args.show_ssim)
            else:
                model.infer(model_dir=args.model_dir, source_obj=args.source_obj, embedding_ids=embedding_ids,
                        save_dir=args.save_dir, progress_file=args.progress_file)
        else:
            if len(embedding_ids) < 2:
                raise Exception("no need to interpolate yourself unless you are a narcissist")
            chains = embedding_ids[:]
            if args.uroboros:
                chains.append(chains[0])
            pairs = list()
            for i in range(len(chains) - 1):
                pairs.append((chains[i], chains[i + 1]))
            for s, e in pairs:
                model.interpolate(model_dir=args.model_dir, source_obj=args.source_obj, between=[s, e],
                                  save_dir=args.save_dir, steps=args.steps)
            if args.output_gif:
                gif_path = os.path.join(args.save_dir, args.output_gif)
                compile_frames_to_gif(args.save_dir, gif_path)
                print("gif saved at %s" % gif_path)


if __name__ == '__main__':
    tf.app.run()

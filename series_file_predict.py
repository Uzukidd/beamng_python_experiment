from utils import *
import carla

from detectors import torch_script_module
from carla_control import carla_client

# from utils.mods.post_processing import mono_label_distance_tracker, multi_classes_assemble_tracker
import numpy as np
import open3d as o3d
import torch
import time

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    import pcdet.config
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network
    from pcdet.utils import common_utils
except Exception as e:
    print(e)
    print("pcdet is not installed")

import argparse


def parse_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-c",
        "--config-filename",
        default="./configs/scale_1_12_predict.yaml",
        help="config filename (*.yaml)",
    )
    argparser.add_argument(
        "-d",
        "--delta-second",
        default=0.1,
        help="delta second between every 2 frames",
    )
    argparser.add_argument(
        "-v",
        "--preview",
        action="store_true",
        help="open an open3d windows to preview current frame",
    )
    argparser.add_argument(
        "-r",
        "--root-path",
        help="root path of the dataset",
    )
    args = argparser.parse_args()
    return args


def model_forwarding(model, idx, data_dict):
    pred_dicts, final_boxes, final_scores, final_labels = None, None, None, None
    before_time = time.perf_counter()

    if data_dict["points"] is not None:
        pred_dicts, ret_dict = model.forward(data_dict)

        if isinstance(pred_dicts, list):
            cls_preds = pred_dicts[0]["pred_scores"]
            box_preds = pred_dicts[0]["pred_boxes"]
            label_preds = pred_dicts[0]["pred_labels"]
        else:
            cls_preds = pred_dicts["pred_scores"]
            box_preds = pred_dicts["pred_boxes"]
            label_preds = pred_dicts["pred_labels"]
            selected, selected_scores = class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds, score_thresh=0.1
            )
            cls_preds = cls_preds[selected]
            label_preds = label_preds[selected]
            box_preds = box_preds[selected]

        final_scores = cls_preds
        final_labels = label_preds
        final_boxes = box_preds
    after_time = time.perf_counter()
    return after_time - before_time, pred_dicts, final_boxes, final_scores, final_labels


def scene_rendering(
    idx,
    points,
    vis,
    final_boxes=None,
    final_scores=None,
    final_labels=None,
    gt_boxes=None,
):
    before_time = time.perf_counter()
    if points is not None:
        draw_scenes(
            vis,
            points=points[:, 1:],
            ref_boxes=final_boxes,
            ref_scores=final_scores,
            ref_labels=final_labels,
            gt_boxes=gt_boxes,
            confidence=None,
        )

    vis.poll_events()
    vis.update_renderer()
    vis.clear_geometries()
    after_time = time.perf_counter()
    return after_time - before_time, None


def main(args):
    logger = create_logger()

    cfg = pcdet.config.cfg_from_yaml_file(args.config_filename, pcdet.config.cfg)
    pcs_dataset = file_point_cloud_dataset(
        dataset_cfg=cfg.DATA_CONFIG,
        logger=logger,
        root_path=args.root_path,
        class_names=cfg.CLASS_NAMES,
    )

    model = None

    try:
        cfg_from_yaml_file("cfgs/kitti_models/pointpillar.yaml", cfg)
        model = build_network(
            model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=pcs_dataset
        )
        model.load_params_from_file(
            filename="D:/modelzoo/pointpillar_7728.pth", logger=logger, to_cpu=True
        )
        model.cuda()
        model.eval()
    except Exception as e:
        print(e)
        logger.error("Failed to load model")

    visualizers = None
    if args.preview:
        visualizers = {}
        view_status = ""
        with open("asset/view_status.json", "r") as input_stream:
            view_status = input_stream.read()

        for channel_name in pcs_dataset.preview_channel:
            visualizers[channel_name] = o3d.visualization.Visualizer()
            visualizers[channel_name].create_window(
                window_name=channel_name, width=1920, height=1080
            )
            visualizers[channel_name].set_view_status(view_status)

    gt_annos = {channel_name: [] for channel_name in pcs_dataset.preview_channel}
    det_annos = {channel_name: [] for channel_name in pcs_dataset.preview_channel}
    try:
        with torch.no_grad():
            for idx, data_series in enumerate(pcs_dataset):
                for channel_name, channel in data_series.items():
                    data_dict = data_series[channel_name]

                    vis = None
                    if visualizers:
                        vis = visualizers[channel_name]

                    ind_gt_annos = gt_annos[channel_name]
                    ind_det_annos = det_annos[channel_name]

                    pre_time = data_dict["pre_time"]
                    forward_time, render_time = 0.0, 0.0
                    final_boxes, final_scores, final_labels = None, None, None
                    gt_boxes = data_dict.get("gt_boxes", None)

                    if data_dict["points"] is not None:
                        batch_dict = pcs_dataset.collate_batch([data_dict])
                        load_data_to_gpu(batch_dict)

                        if model is not None:
                            (
                                forward_time,
                                pred_dicts,
                                final_boxes,
                                final_scores,
                                final_labels,
                            ) = model_forwarding(model, idx, batch_dict)
                            annos = pcs_dataset.generate_prediction_dicts(
                                batch_dict,
                                pred_dicts,
                                cfg.CLASS_NAMES,
                                output_path=None,
                            )

                            ind_gt_annos += [
                                {
                                    "frame_id": data_dict["frame_id"],
                                    "gt_boxes": data_dict.get("gt_boxes", None),
                                }
                            ]
                            ind_det_annos += annos
                        else:
                            forward_time = 0.0
                            final_boxes = None
                            final_scores = None
                            final_labels = None
                        if vis:
                            render_time, _ = scene_rendering(
                                idx,
                                batch_dict["points"],
                                vis,
                                final_boxes,
                                final_scores,
                                final_labels,
                                gt_boxes,
                            )

                        print(
                            f"Compute time: {pre_time:.3f} + {forward_time:.3f} + {render_time:.3f} == {pre_time + forward_time + render_time:.3f}s",
                            end="\r",
                        )
                time.sleep(args.delta_second)

    except KeyboardInterrupt:
        print("prediction interrupted!")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if visualizers:
            for channel_name in pcs_dataset.preview_channel:
                visualizers[channel_name].destroy_window()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

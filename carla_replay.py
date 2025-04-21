from utils import *
import carla

from detectors import torch_script_module
from carla_control import carla_client

# from utils.mods.post_processing import mono_label_distance_tracker, multi_classes_assemble_tracker
import numpy as np
import open3d as o3d
import torch
import time

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
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "-f",
        "--recorder-filename",
        metavar="F",
        default="D:\\project\\scenario_runner\\manual_records\\FollowLeadingVehicleWithObstacle_1.log",
        help="recorder filename (test1.log)",
    )
    argparser.add_argument(
        "-c",
        "--config-filename",
        default=".\\configs\\carla_predict_eval.yaml",
        help="config filename (*.yaml)",
    )
    argparser.add_argument(
        "-v",
        "--preview",
        action="store_true",
        help="open an open3d windows to preview current frame",
    )
    argparser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="evaluate result after finishing replay",
    )
    args = argparser.parse_args()
    return args


class replay_finished(Exception):
    pass


def carla_ticking(client, idx, data_dict):
    before_time = time.perf_counter()

    client.carla_world.tick()

    after_time = time.perf_counter()
    return after_time - before_time, None


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


def evaluation_result(gt_annos, det_annos, class_names: list):
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    import math
    def safe_divide(a: float, b: float) -> float:
        return a / b if b != 0.0 else math.nan

    res_dict = {}
    res_str = ""
    for channel_name in det_annos.keys():
        channel_gt_annos = gt_annos[channel_name]
        channel_det_annos = det_annos[channel_name]

        channel_res = {
            name: {
                "3d": {
                    "recall": 0,
                    "totall": 0,
                }
            }
            for name in class_names
        }
        for ind_gt_annos, ind_det_annos in zip(channel_gt_annos, channel_det_annos):
            gt_boxes, _ = common_utils.check_numpy_to_torch(ind_gt_annos["gt_boxes"])
            det_boxes, _ = common_utils.check_numpy_to_torch(
                ind_det_annos["boxes_lidar"]
            )
            det_label, _ = common_utils.check_numpy_to_torch(
                ind_det_annos["pred_labels"]
            )

            if gt_boxes is None:
                continue
            for label, class_name in enumerate(class_names):
                label = label + 1
                label_gt_boxes = gt_boxes[gt_boxes[:, 7] == label].cuda()
                label_det_boxes = det_boxes[det_label == label].cuda()

                # 3d bounding box metric
                gt_iou3d = iou3d_nms_utils.boxes_iou3d_gpu(
                    label_gt_boxes[:, :7], label_det_boxes
                )
                channel_res[class_name]["3d"]["recall"] += torch.any(
                    gt_iou3d >= 0.7, dim=1
                ).sum().item()
                channel_res[class_name]["3d"]["totall"] += label_gt_boxes.size(0)

        res_dict[channel_name] = channel_res

        res_str += f"-------{channel_name}--------\n"

        for label, class_name in enumerate(class_names):
            res_str += f"{class_name}\tmAP@0.70 NaN NaN\n"
            res_str += f"3d\t mAP:{safe_divide(channel_res[class_name]["3d"]["recall"], channel_res[class_name]["3d"]["totall"]) * 100.0:.2f}\n"

    return res_dict, res_str


def main(args):
    logger = create_logger()
    client = carla_client(host=args.host, port=args.port, logger=logger)
    client.init_client()
    client.start_client()

    cfg = pcdet.config.cfg_from_yaml_file(args.config_filename, pcdet.config.cfg)

    client.replay_file(args.recorder_filename)
    client.connect_to_vehicle("hero")

    pcs_dataset = carla_point_cloud_dataset(
        dataset_cfg=cfg.DATA_CONFIG,
        logger=logger,
        lidar=client.lidar_t,
        class_names=cfg.CLASS_NAMES,
    )
    # object_tracker = multi_classes_assemble_tracker(num_classes=4, track_length=25, multi_head=True)
    # try:
    #     model = torch.jit.load("./torch_scripts/point_pillar_model.pt")
    #     model.cuda()
    #     model.eval()
    # except Exception as e:
    #     logger.error(f"Failed to load jit model from: {e}")
    #     model = None

    model = None

    try:
        # raise NotImplementedError
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
        for channel_name in pcs_dataset.preview_channel:
            visualizers[channel_name] = o3d.visualization.Visualizer()
            visualizers[channel_name].create_window(
                window_name=channel_name, width=1920, height=1080
            )
        # view_control = visualizers[channel_name].get_view_control()
        # view_params = {
        #     "boundingbox_max": np.array([69.118263244628906, 39.679920196533203, 16.415634155273438]),
        #     "boundingbox_min": np.array([-0.059999999999999998, -39.679874420166016, -6.9146575927734375]),
        #     "field_of_view": 60.0,
        #     "front": np.array([-0.90307097537632919, 0.0017988087570628851, 0.42948757574567964]),
        #     "lookat": np.array([34.529131622314452, 2.288818359375e-05, 4.75048828125]),
        #     "up": np.array([0.42948904059539766, 0.0070563614983622357, 0.90304450154510629]),
        #     "zoom": 0.69999999999999996
        # }
        # view_control.set_front(view_params["front"])
        # view_control.set_lookat(view_params["lookat"])
        # view_control.set_up(view_params["up"])
        # view_control.set_zoom(view_params["zoom"])
        # view_control.change_field_of_view(view_params["field_of_view"])
        # view_control.set_constant_z_far(280.0)
    gt_annos = {channel_name: [] for channel_name in pcs_dataset.preview_channel}
    det_annos = {channel_name: [] for channel_name in pcs_dataset.preview_channel}
    try:
        with torch.no_grad():
            for idx, data_series in enumerate(pcs_dataset):
                if client.vehicle.actor_state == carla.ActorState.Invalid:
                    raise replay_finished()

                ticking_time, _ = carla_ticking(client, idx, None)

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

                        # render_time = time.perf_counter()

                        # if final_boxes is not None:
                        #     object_tracker.updates_object(final_boxes[np.newaxis, :, :], final_labels[np.newaxis, :], final_scores[np.newaxis, :])

                        # bounding_boxes, box_scores, box_labels, tracks = object_tracker.get_all_object()

                        # draw_scenes(vis,
                        #     points=data_dict['points'][:, 1:], ref_boxes=bounding_boxes,
                        #     ref_scores=box_scores, ref_labels=box_labels, confidence=None, tracks=tracks
                        # )
                        # draw_scenes(vis,
                        #     points=data_dict['points'][:, 1:], ref_boxes=None,
                        #     ref_scores=None, ref_labels=None, confidence=None, tracks=None
                        # )
                        # vis.poll_events()
                        # vis.update_renderer()
                        # vis.clear_geometries()

                        # render_time = time.perf_counter() - render_time

                        # print(f"Channel name: {channel_name}")
                        print(
                            f"Compute time: {pre_time:.3f} + {ticking_time:.3f} + {forward_time:.3f} + {render_time:.3f} == {pre_time + ticking_time + forward_time + render_time:.3f}s",
                            end="\r",
                        )
                        # print(
                        #     f"Target amount: {len(final_boxes if (final_boxes is not None) else [])}"
                        # )
                        # logger.info(f"current uuid:{object_tracker.get_last_uuid()}")
    except replay_finished:
        print("replay finished!")
    except KeyboardInterrupt:
        print("replay interrupted!")
    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        if visualizers:
            for channel_name in pcs_dataset.preview_channel:
                visualizers[channel_name].destroy_window()
        client.close_client()
        print("CarLA client closed!")

    if args.evaluate:
        res_dict, res_str = evaluation_result(gt_annos, det_annos, cfg.CLASS_NAMES)
        print(res_str)
        import os
        import json

        result_filename = os.path.join(
            "./evaluation_result",
            os.path.splitext(os.path.basename(args.recorder_filename))[0] + ".json",
        )
        with open(result_filename, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=4, ensure_ascii=False, default=str)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

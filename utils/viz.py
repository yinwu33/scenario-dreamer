import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless; every consumer of this module renders to file

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import os
from utils.geometry import *
import math
from cfgs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN
from sim.schema import MIN_DISTANCE_TO_GOAL


def _tensor_to_numpy_for_viz(tensor):
    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.numpy()


# Human-readable names for the adversary's discretized conditioning labels
# (matches dataset_ldm_adv_waymo._adv_condition:
# [type, motion, goal_dist, ego_dist]).
_ADV_COND_TYPE_NAMES = {0: 'vehicle', 1: 'pedestrian', 2: 'cyclist'}
_ADV_COND_MOTION_NAMES = {0: 'parked', 1: 'moving'}
_ADV_COND_DIST_NAMES = {0: 'near', 1: 'middle', 2: 'far'}


def _format_adv_condition_text(cond):
    """Format a single adversary's ``[type, motion, goal_dist, ego_dist]`` labels
    as a readable overlay string, e.g. ``adv: cyclist | parked | goal:far |
    ego:near``. Tolerates the legacy 3-label ``[type, motion, dist]`` form."""
    t = _ADV_COND_TYPE_NAMES.get(int(cond[0]), f"type{int(cond[0])}")
    m = _ADV_COND_MOTION_NAMES.get(int(cond[1]), f"motion{int(cond[1])}")
    goal_d = _ADV_COND_DIST_NAMES.get(int(cond[2]), f"dist{int(cond[2])}")
    if len(cond) > 3:
        ego_d = _ADV_COND_DIST_NAMES.get(int(cond[3]), f"dist{int(cond[3])}")
        return f"adv: {t} | {m} | goal:{goal_d} | ego:{ego_d}"
    return f"adv: {t} | {m} | {goal_d}"


def _draw_state_agent(ax, state, color, V, *, plot_heading_line, zorder, goal_zorder):
    """One agent from a ``[x, y, speed, cos, sin, length, width, goal_x, goal_y]``
    row: box, optional heading line, and goal. Bridges the scene renderer's
    state-array layout onto the shared per-scalar primitives."""
    x, y, length, width = state[0], state[1], state[5], state[6]
    heading = np.arctan2(state[4], state[3])
    _draw_agent_box(ax, x, y, heading, length, width, color, V["bbox_lw"], zorder=zorder)
    if plot_heading_line:
        _draw_heading_line(ax, x, y, heading, length, V["heading_lw"], zorder + 1)
    _draw_goals(ax, state, x, y, color, V, zorder_base=goal_zorder)


def plot_scene(
        agent_states, 
        road_points, 
        agent_types, 
        lane_types, 
        name, 
        save_dir, 
        return_fig=False,
        tile_occupancy=None,
        adaptive_limits=False,
        route=None,
        condition_text=None,
        adv_states=None,
        adv_types=None):
    """Plots a scene with lanes and agents.

    ``adv_states`` (optional, same layout as ``agent_states``) are adversarial
    agents drawn in vivid green on top of the normal agents, matching the DDPO
    rollout convention (``CONTROL_COLOR``)."""

    # Create a figure and axes
    fig, ax = plt.subplots()

    if adaptive_limits:
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        for tile_corners in tile_occupancy:
            x_min = min(x_min, tile_corners[:, 0].min())
            x_max = max(x_max, tile_corners[:, 0].max())
            y_min = min(y_min, tile_corners[:, 1].min())
            y_max = max(y_max, tile_corners[:, 1].max())
    else:
        x_max = 32 
        x_min = -32
        y_max = 32 
        y_min = -32

    V = _view((x_min, x_max), (y_min, y_max))

    _draw_lanes(ax, road_points[..., :2], V, lane_types)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    if condition_text is not None:
        ax.text(
            0.01,
            0.99,
            condition_text,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
            zorder=20,
        )

    # Plot route
    if route is not None:
        ax.plot(route[:, 0], route[:, 1], color='red', linestyle='solid', zorder=5, linewidth=V["route_lw"])

    for a in range(len(agent_states)):
        # nuplan puts immobile objects in the cyclist slot, so they are grey
        # there and amber (a real cyclist) on waymo.
        if lane_types is not None and agent_types[a] == 2:
            color = 'grey'
        else:
            color = _agent_color(a == 0, agent_types[a])
        # A static object has no meaningful facing.
        plot_heading_line = lane_types is None or agent_types[a] in (0, 1)
        _draw_state_agent(ax, agent_states[a], color, V,
                          plot_heading_line=plot_heading_line, zorder=4, goal_zorder=3)

    # The generated adversary sits on top of the normal agents, and green means
    # only this in a frame -- see CONTROL_COLOR.
    if adv_states is not None:
        for a in range(len(adv_states)):
            plot_heading_line = (
                lane_types is None or adv_types is None or adv_types[a] in (0, 1)
            )
            _draw_state_agent(ax, adv_states[a], CONTROL_COLOR, V,
                              plot_heading_line=plot_heading_line, zorder=9, goal_zorder=10)

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if return_fig:
        # Return the figure object for logging
        return fig
    else:
        plt.margins(0)  # Remove margins
        ax.margins(0)  # Ensure no margins in axes
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Fill the entire figure canvas
        
        # Save the figure without margins
        fig.savefig(
            os.path.join(save_dir, name),
            dpi=1000,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)
        return None


def plot_lane_graph(
        road_points, 
        lane_conn, 
        edge_index_lane_to_lane, 
        lane_conn_type, 
        name, 
        save_dir, 
        return_fig=False):
    """
    Plots a lane graph with road points and semantic connections between lanes."""
    
    # Create a figure and axes
    fig, ax = plt.subplots()

    ct = 0
    for i in range(len(road_points)):
        lane = road_points[i, :, :2]
        
        ax.plot(lane[:, 0], lane[:, 1], color='black', linewidth=1.5)
        ct += 1

        label_idx = len(lane) // 2
        ax.annotate(
            str(i),
            (lane[label_idx, 0], lane[label_idx, 1]),
            zorder=5,
            fontsize=5
        )

    for j in range(lane_conn.shape[0]):
        if lane_conn[j, lane_conn_type] == 1:
            src_idx = edge_index_lane_to_lane[0, j]
            dest_idx = edge_index_lane_to_lane[1, j]
            
            lane_src = road_points[src_idx, :, :2]
            lane_dest = road_points[dest_idx, :, :2]
            src_pos = lane_src[10, :2]
            dest_pos = lane_dest[10, :2]

            if lane_conn.shape[1] == 6:
                edge_color = 'purple'
                if lane_conn[j, 2] == 1:
                    edge_color = 'red'
                elif lane_conn[j, 3] == 1:
                    edge_color = 'green'
                elif lane_conn[j, 4] == 1:
                    edge_color = 'blue'
            else:
                if lane_conn[j, 1] == 1:
                    edge_color = 'red'
                elif lane_conn[j, 2] == 1:
                    edge_color = 'green'

            ax.arrow(
                src_pos[0], src_pos[1],
                dest_pos[0] - src_pos[0], dest_pos[1] - src_pos[1],
                length_includes_head=True,
                head_width=1, head_length=1,
                zorder=10, color=edge_color
            )

    # Adjust plot settings if needed
    ax.set_aspect('equal', adjustable='box')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if return_fig:
        # Return the figure object for logging
        return fig
    else:
        # Save the figure to disk and close it
        fig.savefig(os.path.join(save_dir, name), dpi=1000)
        plt.close(fig)
        return None
    

def visualize_batch(num_samples,
                    agent_samples,
                    lane_samples,
                    agent_types,
                    lane_types,
                    lane_conn_samples,
                    data,
                    save_dir,
                    epoch,
                    batch_idx,
                    save_wandb=False,
                    visualize_lane_graph=False,
                    tag='scene_plot',
                    adv_samples=None,
                    adv_batch=None,
                    adv_types=None,
                    adv_cond=None):
    """ Visualize samples from the batch.

    ``tag`` namespaces the saved filenames and W&B panel keys so multiple calls
    (e.g. generated samples vs. ground truth) don't overwrite each other.

    ``adv_samples`` (optional, with per-node ``adv_batch`` scene indices and
    ``adv_types``) are adversarial agents drawn in green on each scene plot.

    ``adv_cond`` (optional, ``[batch_size, 4]`` of discretized
    [type, motion, goal_dist, ego_dist] labels) is written as a text overlay on
    each scene plot.
    """

    if lane_conn_samples.shape[-1] == 4:
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_NUPLAN
    else:
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_WAYMO

    agent_samples = _tensor_to_numpy_for_viz(agent_samples)
    lane_samples = _tensor_to_numpy_for_viz(lane_samples)
    agent_types = _tensor_to_numpy_for_viz(agent_types)
    if lane_types is not None:
        lane_types = _tensor_to_numpy_for_viz(lane_types)
    lane_conn_samples = _tensor_to_numpy_for_viz(lane_conn_samples)
    if adv_samples is not None:
        adv_samples = _tensor_to_numpy_for_viz(adv_samples)
        adv_batch = _tensor_to_numpy_for_viz(adv_batch)
        if adv_types is not None:
            adv_types = _tensor_to_numpy_for_viz(adv_types)
    if adv_cond is not None:
        adv_cond = _tensor_to_numpy_for_viz(adv_cond)

    # pyg data structures for indexing
    lane_batch = data['lane'].batch
    lane_row = data['lane', 'to', 'lane'].edge_index[0]
    lane_conn_batch = lane_batch[lane_row]
    edge_index_l2l = data['lane', 'to', 'lane'].edge_index
    lane_conn_batch = lane_conn_batch.cpu().numpy()
    agent_batch = data['agent'].batch.cpu().numpy()
    lane_batch = data['lane'].batch.cpu().numpy()
    condition_texts = None
    if 'condition_raw' in data.keys():
        condition_raw = _tensor_to_numpy_for_viz(data['condition_raw'])
        condition_clipped = _tensor_to_numpy_for_viz(data['condition_clipped'])
        condition_texts = []
        for condition_idx in range(condition_raw.shape[0]):
            raw = condition_raw[condition_idx]
            clipped = condition_clipped[condition_idx]
            if not np.isclose(raw[0], clipped[0]):
                junction_text = f"junctions={raw[0]:.1f} (clip {clipped[0]:.1f})"
            else:
                junction_text = f"junctions={raw[0]:.1f}"
            condition_texts.append(f"{junction_text}\ncurvature={raw[1]:.3f}")

    import wandb  # heavy; only the save_wandb path needs it

    images_to_log = {}
    for i in range(num_samples):
        # plot the scene
        scene_i_agents = agent_samples[agent_batch == i]
        scene_i_lanes = lane_samples[lane_batch == i]
        scene_i_agent_types = agent_types[agent_batch == i]
        if lane_types is not None:
            scene_i_lane_types = lane_types[lane_batch == i]
        else:
            scene_i_lane_types = None
        if adv_samples is not None:
            scene_i_adv = adv_samples[adv_batch == i]
            scene_i_adv_types = adv_types[adv_batch == i] if adv_types is not None else None
        else:
            scene_i_adv = None
            scene_i_adv_types = None

        # Compose the per-scene condition overlay text: existing map condition (if
        # any) plus the adversary's discretized [type, motion, dist] labels.
        scene_condition_text = condition_texts[i] if condition_texts is not None else None
        if adv_cond is not None and adv_batch is not None:
            scene_i_adv_cond = adv_cond[adv_batch == i]
            if len(scene_i_adv_cond) > 0:
                adv_text = _format_adv_condition_text(scene_i_adv_cond[0])
                scene_condition_text = adv_text if scene_condition_text is None else f"{scene_condition_text}\n{adv_text}"

        fig = plot_scene(
            scene_i_agents,
            scene_i_lanes,
            scene_i_agent_types,
            scene_i_lane_types,
            name=f'{tag}_epoch_{epoch}_batch_{batch_idx}_sample_{i}.png',
            save_dir=save_dir,
            return_fig=save_wandb,
            condition_text=scene_condition_text,
            adv_states=scene_i_adv,
            adv_types=scene_i_adv_types)
        if save_wandb:
            images_to_log[f'{tag}/batch_{batch_idx}_sample_{i}'] = wandb.Image(fig)
            plt.close(fig)

        # plot the lane graph for each edge type
        if visualize_lane_graph:
            scene_i_lane_conns = lane_conn_samples[lane_conn_batch == i]
            shift = np.where(lane_batch == i)[0].min()
            edge_index_i_l2l = edge_index_l2l[:, lane_conn_batch == i].cpu().numpy() - shift
            # {"none": 0, "pred": 1, "succ": 2, "self": 3} (no left/right connections)
            if lane_conn_samples.shape[-1] == 4:
                edge_type_list = [LANE_CONNECTION_TYPES['pred'], LANE_CONNECTION_TYPES['succ']]
            else:
                edge_type_list = [LANE_CONNECTION_TYPES['pred'], LANE_CONNECTION_TYPES['succ'],
                                LANE_CONNECTION_TYPES['left'], LANE_CONNECTION_TYPES['right']]

            for typ in edge_type_list:
                fig = plot_lane_graph(
                    scene_i_lanes, 
                    scene_i_lane_conns, 
                    edge_index_i_l2l, 
                    typ, 
                    name=f'epoch_{epoch}_batch_{batch_idx}_sample_{i}_lanegraph_{typ}.png', 
                    save_dir=save_dir,
                    return_fig=save_wandb)
                if save_wandb:
                    images_to_log[f'lane_graph/batch_{batch_idx}_sample_{i}_type_{typ}'] = wandb.Image(fig)
                    plt.close(fig)
    
    # Log all images at once
    if save_wandb:
        return images_to_log
    else:
        return None


def visualize_predicted_graph(num_samples,
                              agent_samples,
                              agent_batch,
                              agent_types,
                              pred_lanes,
                              pred_lane_batch,
                              save_dir,
                              epoch,
                              batch_idx,
                              save_wandb=False,
                              tag='scene_plot_pred'):
    """Visualize the *threshold-built* predicted lane graph (no GT matching).

    ``pred_lanes`` / ``pred_lane_batch`` come from
    ``AutoEncoderBezier.reconstruct_graph`` -- a variable number of lanes per
    scene built purely from predicted node/edge existence. Agents are reused
    from the GT-aligned reconstruction for context. Logged under ``tag`` so it
    sits next to the GT-matched ``scene_plot`` panels in W&B.
    """
    agent_samples = _tensor_to_numpy_for_viz(agent_samples)
    agent_types = _tensor_to_numpy_for_viz(agent_types)
    agent_batch = _tensor_to_numpy_for_viz(agent_batch)
    pred_lanes = _tensor_to_numpy_for_viz(pred_lanes)
    pred_lane_batch = _tensor_to_numpy_for_viz(pred_lane_batch)

    import wandb  # heavy; only the save_wandb path needs it

    images_to_log = {}
    for i in range(num_samples):
        scene_i_agents = agent_samples[agent_batch == i]
        scene_i_agent_types = agent_types[agent_batch == i]
        scene_i_lanes = pred_lanes[pred_lane_batch == i]
        if scene_i_lanes.shape[0] == 0:  # no edge passed the threshold for this scene
            scene_i_lanes = np.zeros((0, pred_lanes.shape[1] if pred_lanes.ndim == 3 else 20, 2))
        fig = plot_scene(
            scene_i_agents,
            scene_i_lanes,
            scene_i_agent_types,
            None,
            name=f'{tag}_epoch_{epoch}_batch_{batch_idx}_sample_{i}.png',
            save_dir=save_dir,
            return_fig=save_wandb)
        if save_wandb:
            images_to_log[f'{tag}/batch_{batch_idx}_sample_{i}'] = wandb.Image(fig)
            plt.close(fig)

    if save_wandb:
        return images_to_log
    return None


def plot_k_disks_vocabulary(V, png_path, dpi=1000):
    plt.figure(figsize=(18, 3))
    plt.ylim(-0.25, 0.25)
    for state in V:
        plt.scatter(state[0], state[1], s=1, color='blue')
        arrow_length = 0.08  # Define length of arrows
        dx = arrow_length * np.cos(state[2])  # Change in x
        dy = arrow_length * np.sin(state[2])  # Change in y
        plt.plot([state[0], state[0] + dx], [state[1], state[1] + dy], linewidth=0.5, color="black")
    plt.savefig(png_path, dpi=dpi)
    plt.clf()


def render_state(
        agent_states, 
        agent_types, 
        route, 
        lanes, 
        lanes_mask, 
        t, 
        name, 
        movie_path='video_frames', 
        lightweight=False
    ):
    """ Renders the current state of the simulation and saves it as a PNG image."""
    png_dir = f'{movie_path}/{name}'
    if not os.path.exists(png_dir):
        os.makedirs(png_dir, exist_ok=True)

    agent_alpha = 1.0
    agent_zord = 4
    ego_color = '#de5959'
    ego_alpha = 1.0
    ego_zord = 5

    x_min, y_min, x_max, y_max = -75, -75, 75, 75

    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    lanes = np.concatenate([lanes, lanes_mask[:, :, None]], axis=-1)

    # Plot lanes only
    color = 'grey'
    linestyle = 'dashed'
    zorder = 2
    for lane in lanes:
        exists = lane[:, 2] == 1
        plt.plot(
            lane[:, 0][exists], 
            lane[:, 1][exists], 
            color=color, 
            linewidth=1.5, 
            linestyle=linestyle, 
            zorder=zorder
        )
        plt.plot(
            lane[:, 0][exists], 
            lane[:, 1][exists], 
            color='lightgrey', 
            linewidth=20, 
            linestyle='solid', 
            zorder=zorder-1
        )
        if exists[0]:
            plt.scatter(
                lane[0, 0], 
                lane[0, 1], 
                color=color, 
                s=8, 
                zorder=zorder+1
            )
        if exists[-1]:
            plt.scatter(
                lane[-1, 0], 
                lane[-1, 1], 
                color=color, 
                s=8, 
                zorder=zorder+1
            )

    agent_types = np.argmax(agent_types, axis=1)
    
    # Plot agent bounding boxes and headings
    for a in range(len(agent_states)):
        if agent_states[a, -1] == 0:
            continue

        edgecolor = 'black'
        if a == len(agent_states) - 1:
            color = ego_color 
            alpha = ego_alpha 
            zord = ego_zord
        else:
            alpha = agent_alpha 
            zord = agent_zord
            
            if agent_types[a] == 1:
                color = '#87b3e6' # Light blue
            elif agent_types[a] == 2:
                color = '#bea9f5' # Light purple
            elif agent_types[a] == 3:
                color = 'grey'
            else:
                color = "grey"

        # Draw bounding boxes
        length = agent_states[a, 5] * 0.8
        width = agent_states[a, 6] * 0.8
        bbox_x_min = agent_states[a, 0] - width / 2
        bbox_y_min = agent_states[a, 1] - length / 2
        lw = 0.35 / ((x_max - x_min) / 140)
        rectangle = mpatches.FancyBboxPatch(
            (bbox_x_min, bbox_y_min), 
            width, 
            length, 
            ec=edgecolor, 
            fc=color,
            linewidth=lw, 
            alpha=alpha, 
            boxstyle=mpatches.BoxStyle("Round", pad=0.3), 
            zorder=zord
        )
        
        tr = transforms.Affine2D().rotate_deg_around(
            agent_states[a, 0], 
            agent_states[a, 1], 
            np.degrees(agent_states[a, 4]) - 90
        ) + ax.transData
        rectangle.set_transform(tr)
        ax.add_patch(rectangle)
        
        # Draw heading line
        if agent_types[a] in [1, 2]:
            heading_length = length / 2 + 1.5
            heading_angle_rad = agent_states[a, 4]
            vehicle_center = agent_states[a, :2]
            line_end_x = (vehicle_center[0] + 
                          heading_length * math.cos(heading_angle_rad))
            line_end_y = (vehicle_center[1] + 
                          heading_length * math.sin(heading_angle_rad))
            ax.plot(
                [vehicle_center[0], line_end_x], 
                [vehicle_center[1], line_end_y], 
                color='black', 
                zorder=zord+1, 
                alpha=0.25, 
                linewidth=0.3 / ((x_max - x_min) / 140))
    
    # for debugging
    # ax.annotate(a, (vehicle_center[0], vehicle_center[1]), zorder=8, fontsize=5) 
    
    if route is not None:
        plt.scatter(
            route[:, 0], 
            route[:, 1], 
            color=ego_color, 
            zorder=ego_zord, 
            s=8
        )
    plt.tight_layout()
    dpi = 100 if lightweight else 500
    plt.savefig(f'{png_dir}/frame_{t:03}.png', dpi=dpi)
    plt.close(fig)


def generate_video(name, output_dir, delete_images=False):
    """ Generates a video from a sequence of images saved in a directory."""
    image_folder = f'{output_dir}/{name}'
    
    # Get list of all image files in the directory
    images = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    images = [str1.replace('\n', '') for str1 in images]
    images.sort()  # Sort by filename

    # Create a video clip from the image sequence
    from moviepy.editor import ImageSequenceClip  # heavy; only this path needs it

    clip = ImageSequenceClip(images, fps=20)
    
    # Write the video file
    clip.write_videofile(f"{image_folder}.mp4", codec='libx264')

    if delete_images:
        for image in images:
            os.remove(image)


# ---------------------------------------------------------------------- rollout rendering
_EGO_COLOR = "#de5959"      # light red  (ego = local index 0)
_VEH_COLOR = "#87b3e6"      # light blue (other vehicles)
_PED_COLOR = "#bea9f5"      # light purple (pedestrians)
_CYC_COLOR = "#e8b800"      # amber (cyclists); deliberately NOT green, so the
                            # only green in a frame is CONTROL_COLOR below
CONTROL_COLOR = "#2ca02c"   # vivid green: DDPO-generated non-ego agents,
                            # passed in via ``agent_colors`` to flag who is being trained
_JUMP_THRESH = 10.0         # metres/step above which motion is a teleport, not driving
_PARKING_DIST = MIN_DISTANCE_TO_GOAL  # goal within this of spawn => parked/static
FOV = 64.0                  # generated field of view (metres); the view window is
                            # pinned to this square (centred at 0) so anything that
                            # leaves the 64x64 FOV is clipped out of frame. Agents
                            # are only removed from the sim at ``map_extent``, which
                            # may be larger, so the clip and the removal differ.


def _agent_color(is_ego: bool, type_id) -> str:
    if is_ego:
        return _EGO_COLOR
    return {0: _VEH_COLOR, 1: _PED_COLOR, 2: _CYC_COLOR}.get(int(type_id), "grey")


def _agent_draw_color(a: int, is_ego: bool, agent_types, agent_colors) -> str:
    if agent_colors is not None and a < len(agent_colors) and agent_colors[a] is not None:
        return agent_colors[a]
    return _agent_color(is_ego, agent_types[a] if agent_types is not None else 0)


def _first_episode_end(done) -> int | None:
    if done is None or len(done) == 0 or not np.any(done):
        return None
    return int(np.argmax(done)) + 1  # include the last pre-reset state


def _respawn_mask(traj, a, end=None):
    """Per-step bool mask (True once agent ``a`` is inactive) or None if unavailable.

    Respawned or removed agents should no longer be drawn as normal traffic after
    goal arrival. Sliced to ``[:end]`` to match the trajectory slice."""
    ra = traj.get("respawn") if isinstance(traj, dict) else None
    if ra is None or getattr(ra, "ndim", 0) != 2 or ra.shape[1] <= a:
        return None
    return ra[:end, a] if end is not None else ra[:, a]


def _break_on_jumps(x, y):
    if len(x) < 2:
        return x, y
    bad = np.where(np.hypot(np.diff(x), np.diff(y)) > _JUMP_THRESH)[0]
    if len(bad) == 0:
        return x, y
    return np.insert(x.astype(float), bad + 1, np.nan), np.insert(y.astype(float), bad + 1, np.nan)


def _fmt_float(value, *, signed: bool = False, digits: int = 2, inf: str = "inf") -> str:
    if value is None:
        return "nan"
    v = float(value)
    if not np.isfinite(v):
        return inf
    sign = "+" if signed else ""
    return f"{v:{sign}.{digits}f}"


# Reward-component breakdown rendered (one list == one line, in order) under the
# summary line when the caller passes a ``components`` dict (e.g. from
# RewardModel.evaluate). Each field is (short label, component key); missing
# keys are skipped so this degrades gracefully as new components are added. The
# leading field of each line is that line's total. Layout mirrors the reward
# assembly: constraint + its penalty terms, then criticality + its terms, then the
# raw ego<->adversary / lane geometry behind those terms.
_COMPONENT_LINES = [
    # Which band the scene landed in and what put it there. `tier` is the band
    # index the reward assigned (hierarchical_v6: 0 invalid, 1 d_min, 2 near
    # miss, 3 ego-fault collision, 4 uncredited ram); the flags beside it are
    # the mutually exclusive reasons a scene can be rejected or uncredited, so
    # the number is always traceable to a cause.
    [("tier", "tier"), ("coll", "r_collision"), ("ram", "c_rammed"),
     ("offlane", "c_offlane"), ("early", "c_trivial"), ("cond", "c_invalid")],
    # The two graded quantities the credited bands are built from.
    [("minTTCego", "ego_min_ttc"), ("dmin", "ego_adv_min_dist_warmup"),
     ("g_ttc", "r_ttc"), ("g_d", "r_approach"), ("d0", "ego_adv_init_dist")],
    # Raw geometry behind the rejections.
    [("spawn_lane", "spawn_lane_dist"), ("goal_lane", "goal_lane_dist"),
     ("overlap", "init_overlap_frac")],
]


def _status_text(
    reward,
    collided,
    init_invalid,
    *,
    ego_min_ttc=None,
    goal_offlane_frac=None,
    parking_mismatch_frac=None,
    components=None,
) -> tuple[str, str]:
    """Reward status line(s) aligned with the DDPO reward formula.

    ``components`` (optional) is a per-scene mapping of reward-component name ->
    scalar; when present, the full criticality / constraint / geometry breakdown
    is rendered on extra lines below the summary.
    """
    r = 0.0 if reward is None else float(reward)
    if components:
        # Summary line: total reward + the hard reject branch flag (a parked /
        # condition-violating adversary is rejected outright, reward = -1), then
        # the component lines. c_invalid (condition check) supersedes c_parking
        # when present; fall back to the parking flag otherwise.
        reject_val = components.get("c_invalid", components.get("c_parking", 0.0))
        reject_reason = components.get("c_invalid_reason", "")
        park = bool(float(reject_val) > 0.0)
        reject_reason = str(reject_reason).strip()
        reason_txt = f"  reason={reject_reason}" if reject_reason else ""
        lines = [f"R={r:+.2f}  reject={str(park).lower()}{reason_txt}"]
        for fields in _COMPONENT_LINES:
            parts = [
                f"{short}={_fmt_float(components[key])}"
                for short, key in fields
                if key in components
            ]
            if parts:
                lines.append("  ".join(parts))
    else:
        # No components supplied: the reward's own breakdown is unavailable, so
        # report the rollout facts the hierarchical rewards are built on.
        # parking_mismatch is deliberately absent -- it enters neither `total`
        # nor `invalid` in v3 and later, so it only crowded the line.
        lines = [
            f"R={r:+.2f}  minTTCego={_fmt_float(ego_min_ttc)}  "
            f"coll={int(bool(collided))}  inval={int(bool(init_invalid))}  "
            f"goal_off={_fmt_float(goal_offlane_frac)}"
        ]
    if collided:
        color = _EGO_COLOR
    elif init_invalid:
        color = "#ff7f0e"
    else:
        color = "0.3"
    return "\n".join(lines), color


def _draw_agent_box(ax, x, y, heading, length, width, color, lw, alpha=1.0,
                    zorder=6, edgecolor="black"):
    """Rounded width(x)*length(y) box centred at (x,y), rotated by degrees(heading)-90."""
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(heading)):
        return  # blanked-out step (e.g. post-respawn): nothing to draw
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, y - length / 2), width, length,
        ec=edgecolor, fc=color, linewidth=lw, alpha=alpha,
        boxstyle=mpatches.BoxStyle("Round", pad=0.3), zorder=zorder,
    )
    tr = transforms.Affine2D().rotate_deg_around(x, y, np.degrees(heading) - 90) + ax.transData
    rect.set_transform(tr)
    ax.add_patch(rect)


def _view(xlim=None, ylim=None):
    """Fixed FOV-sized square view window centred at the scene origin.

    The window is pinned to the generated field of view (``FOV`` metres, centred
    at 0) regardless of where agents drive, so anything that leaves the 64x64 FOV
    is clipped out of frame. Linewidths keep the same 64 m reference as before
    (``scale == 1`` here)."""
    if xlim is None:
        half = FOV / 2.0
        xlim = ylim = (-half, half)
    scale = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 64.0
    return {
        "xlim": xlim, "ylim": ylim,
        "base_lw": 1.5 / scale, "road_w": 20.0 / scale, "scatter": 8.0 / (scale ** 2),
        "bbox_lw": 0.35 / scale, "goal_lw": 0.6 / scale, "goal_ms": 28.0 / (scale ** 2),
        "heading_lw": 0.3 / scale, "route_lw": 1.5 / scale,
    }


def _draw_lanes(ax, lanes_arr, V, lane_types=None):
    """Centerlines, plus the wide light-grey stroke that stands in for the road.

    ``lane_types`` is the nuplan/waymo lane class per polyline: 0 centerline,
    1 green traffic light, 2 red. Only a centerline gets the road stroke, and a
    traffic-light lane is drawn in its own colour one layer up. Omit it (the
    rollout case) and every polyline is treated as a centerline."""
    if lanes_arr is None:
        return
    for i, poly in enumerate(lanes_arr):
        pts = poly[np.isfinite(poly[:, 0]) & np.isfinite(poly[:, 1])]
        if len(pts) < 2:
            continue
        kind = 0 if lane_types is None else int(lane_types[i])
        color = {0: "grey", 1: "green"}.get(kind, "red")
        zorder = 2 if kind == 0 else 3
        if kind == 0:
            ax.plot(pts[:, 0], pts[:, 1], color="lightgrey", linewidth=V["road_w"],
                    linestyle="solid", zorder=zorder - 1)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=V["base_lw"],
                linestyle="dashed", zorder=zorder)
        ax.scatter(pts[[0, -1], 0], pts[[0, -1], 1], color=color, s=V["scatter"],
                   zorder=zorder + 1)


def _draw_parked_marker(ax, x, y, V, zorder):
    """Bold black cross at a parked agent's centre; see _draw_goals."""
    ax.scatter(x, y, marker="x", color="black", s=V["goal_ms"],
               linewidths=max(V["goal_lw"] * 2.0, 1.2), zorder=zorder)


def _draw_goals(ax, agent_states, x0, y0, color, V, zorder_base=3):
    if agent_states is None or agent_states.shape[0] < 9:
        return
    gx, gy = float(agent_states[7]), float(agent_states[8])
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return
    if np.hypot(gx - x0, gy - y0) < _PARKING_DIST:
        # Parked/static agent: the goal sits on the spawn, so there is no travel
        # to draw. A cross, not a ring -- the two markers then say different
        # things at a glance: a ring is somewhere the agent is going, a cross is
        # an agent that is staying put.
        _draw_parked_marker(ax, x0, y0, V, zorder_base + 4)
        return
    ax.plot([x0, gx], [y0, gy], color=color, linestyle=":", alpha=0.7,
            linewidth=V["goal_lw"], zorder=zorder_base)
    _draw_goal_target(ax, gx, gy, color, V["goal_lw"], zorder_base + 4)


# Goal target geometry in metres, so it reads the same at any view scale.
_GOAL_RING_RADIUS = 2.0
_GOAL_DOT_RADIUS = 0.30


def _draw_goal_target(ax, x, y, color, lw, zorder=7):
    """A hollow ring with a filled centre dot, in data (metre) units."""
    ax.add_patch(mpatches.Circle(
        (x, y), radius=_GOAL_RING_RADIUS, fill=False, edgecolor=color,
        linewidth=max(lw, 0.7), zorder=zorder,
    ))
    ax.add_patch(mpatches.Circle(
        (x, y), radius=_GOAL_DOT_RADIUS, facecolor=color, edgecolor="none",
        zorder=zorder + 1,
    ))


def _finish(ax, fig, V, title, status_txt, status_color, *, annotate=True):
    ax.set_xlim(*V["xlim"]); ax.set_ylim(*V["ylim"])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if annotate:
        ax.set_title(f"{title}\n{status_txt}", fontsize=8.5, color=status_color)
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


def render_rollout(traj, lanes, *, agent_states=None, agent_types=None, agent_colors=None,
                   reward=None, ego_collision=False, ego_offroad=False, init_invalid=False,
                   ego_min_ttc=None, goal_offlane_frac=None, parking_mismatch_frac=None,
                   components=None, title="", final_boxes_only=False,
                   annotate=True) -> "plt.Figure":
    """Static summary of the first episode with full, per-agent-coloured trajectories."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    x, y, hd = traj["x"], traj["y"], traj["heading"]
    n_agents = x.shape[1] if (x.ndim == 2 and x.size) else 0
    end = _first_episode_end(traj.get("done"))
    lanes_arr = np.asarray(lanes) if (lanes is not None and len(lanes) > 0) else None
    V = _view()
    _draw_lanes(ax, lanes_arr, V)

    n_boxes = 5
    for a in range(n_agents):
        is_ego = a == 0
        color = _agent_draw_color(a, is_ego, agent_types, agent_colors)
        xa, ya, ha = x[:end, a], y[:end, a], hd[:end, a]
        if len(xa) == 0:
            continue
        # Blank out inactive steps so we don't draw the agent as normal traffic.
        mask = _respawn_mask(traj, a, end)
        if mask is not None and mask.any():
            xa = np.where(mask, np.nan, xa)
            ya = np.where(mask, np.nan, ya)
        length, width = float(traj["length"][a]), float(traj["width"][a])
        xb, yb = _break_on_jumps(xa, ya)
        trajectory_scale = (
            (3.2 if is_ego else 2.6) if final_boxes_only else (1.6 if is_ego else 1.1)
        )
        trajectory_alpha = 1.0 if final_boxes_only else (0.95 if is_ego else 0.7)
        ax.plot(xb, yb, color=color, linewidth=V["base_lw"] * trajectory_scale,
                alpha=trajectory_alpha, zorder=5 if is_ego else 4, solid_capstyle="round")
        if not final_boxes_only:
            ax.scatter(xa, ya, color=color, s=V["scatter"] * 0.35,
                       alpha=0.9 if is_ego else 0.6,
                       zorder=5 if is_ego else 4, edgecolors="none")
        idxs = np.array([len(xa) - 1]) if final_boxes_only else np.unique(
            np.linspace(0, len(xa) - 1, min(n_boxes, len(xa))).round().astype(int)
        )
        for j, t in enumerate(idxs):
            frac = (j + 1) / len(idxs)
            _draw_agent_box(ax, xa[t], ya[t], ha[t], length, width, color,
                            V["bbox_lw"] * (1.4 if t == idxs[-1] else 1.0), alpha=0.22 + 0.78 * frac)
        _draw_goals(ax, agent_states[a] if agent_states is not None else None, xa[0], ya[0], color, V)

    txt, scol = _status_text(
        reward,
        ego_collision,
        init_invalid,
        ego_min_ttc=ego_min_ttc,
        goal_offlane_frac=goal_offlane_frac,
        parking_mismatch_frac=parking_mismatch_frac,
        components=components,
    )
    _finish(ax, fig, V, title, txt, scol, annotate=annotate)
    return fig


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3].copy()


def render_rollout_frames(traj, lanes, *, agent_states=None, agent_types=None, agent_colors=None,
                          reward=None, ego_collision=False, ego_offroad=False, init_invalid=False,
                          ego_min_ttc=None, goal_offlane_frac=None, parking_mismatch_frac=None,
                          components=None, title="", max_frames=50, annotate=True) -> np.ndarray:
    """One frame per rollout step (agents move, trail grows). Returns [T, H, W, 3] uint8.

    The view window and reward text are fixed across frames so the GIF is stable.
    """
    x, y, hd = traj["x"], traj["y"], traj["heading"]
    n_agents = x.shape[1] if (x.ndim == 2 and x.size) else 0
    end = _first_episode_end(traj.get("done"))
    T = end if end is not None else (x.shape[0] if n_agents else 0)
    T = max(T, 1)
    lanes_arr = np.asarray(lanes) if (lanes is not None and len(lanes) > 0) else None
    V = _view()
    txt, scol = _status_text(
        reward,
        ego_collision,
        init_invalid,
        ego_min_ttc=ego_min_ttc,
        goal_offlane_frac=goal_offlane_frac,
        parking_mismatch_frac=parking_mismatch_frac,
        components=components,
    )
    lengths = [float(traj["length"][a]) for a in range(n_agents)]
    widths = [float(traj["width"][a]) for a in range(n_agents)]

    frame_ts = np.unique(np.linspace(0, T - 1, min(max_frames, T)).round().astype(int))
    frames = []
    for t in frame_ts:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        _draw_lanes(ax, lanes_arr, V)
        for a in range(n_agents):
            is_ego = a == 0
            color = _agent_draw_color(a, is_ego, agent_types, agent_colors)
            mask = _respawn_mask(traj, a, t + 1)
            if mask is not None and bool(mask[-1]):
                continue  # agent is inactive by now: drop it instead of faking traffic
            xa, ya, ha = x[:t + 1, a], y[:t + 1, a], hd[:t + 1, a]
            if mask is not None and mask.any():
                xa = np.where(mask, np.nan, xa)
                ya = np.where(mask, np.nan, ya)
            xb, yb = _break_on_jumps(xa, ya)
            ax.plot(xb, yb, color=color, linewidth=V["base_lw"] * (1.3 if is_ego else 0.9),
                    alpha=0.5, zorder=4, solid_capstyle="round")  # trail so far
            _draw_agent_box(ax, x[t, a], y[t, a], hd[t, a], lengths[a], widths[a], color,
                            V["bbox_lw"] * (1.4 if is_ego else 1.0), alpha=0.8)  # current pose
            _draw_goals(ax, agent_states[a] if agent_states is not None else None, x[0, a], y[0, a], color, V)
        _finish(ax, fig, V, f"{title}   t={int(t)}", txt, scol, annotate=annotate)
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)
    return np.stack(frames, axis=0)


def save_gif(frames: np.ndarray, path: str, fps: int = 10) -> str:
    """Write [T,H,W,3] uint8 frames to an animated GIF via Pillow (no moviepy dep)."""
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0, optimize=True)
    return path


def _draw_heading_line(ax, x, y, heading, length, lw, zorder):
    """Short black stick out of an agent's nose, so a still shows facing."""
    reach = length / 2 + 1.5
    ax.plot([x, x + reach * math.cos(heading)], [y, y + reach * math.sin(heading)],
            color="black", alpha=0.5, linewidth=lw, zorder=zorder)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import os
from utils.geometry import *
import math
from cfgs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN
from ddpo.goal_schema import MIN_DISTANCE_TO_GOAL
from ddpo.viz import CONTROL_COLOR  # vivid green for adversarial/controlled agents
from moviepy.editor import ImageSequenceClip
import wandb


def _tensor_to_numpy_for_viz(tensor):
    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.numpy()


# Human-readable names for the adversary's discretized conditioning labels
# (matches dataset_ldm_adv_waymo._adv_condition: [type, motion, dist]).
_ADV_COND_TYPE_NAMES = {0: 'vehicle', 1: 'pedestrian', 2: 'cyclist'}
_ADV_COND_MOTION_NAMES = {0: 'parked', 1: 'moving'}
_ADV_COND_DIST_NAMES = {0: 'near', 1: 'middle', 2: 'far'}


def _format_adv_condition_text(cond):
    """Format a single adversary's ``[type, motion, dist]`` label triple as a
    readable overlay string, e.g. ``adv: cyclist | parked | far``."""
    t = _ADV_COND_TYPE_NAMES.get(int(cond[0]), f"type{int(cond[0])}")
    m = _ADV_COND_MOTION_NAMES.get(int(cond[1]), f"motion{int(cond[1])}")
    d = _ADV_COND_DIST_NAMES.get(int(cond[2]), f"dist{int(cond[2])}")
    return f"adv: {t} | {m} | {d}"


def _draw_agent_box(ax, state, color, bbox_linewidth, heading_linewidth,
                    plot_heading_line, alpha=1.0, edgecolor='black', zorder=4):
    """Draw a single rounded agent bounding box (and optional heading line).

    ``state`` is the ``[x, y, speed, cosθ, sinθ, length, width]`` agent layout.
    The heading line is drawn one zorder above the box. Shared by the normal
    agents and the adversarial agent so both render identically (only the colour
    and zorder differ)."""
    length = state[5]
    width = state[6]
    bbox_x_min = state[0] - width / 2
    bbox_y_min = state[1] - length / 2
    rectangle = mpatches.FancyBboxPatch(
        (bbox_x_min, bbox_y_min),
        width, length,
        ec=edgecolor, fc=color,
        linewidth=bbox_linewidth, alpha=alpha,
        boxstyle=mpatches.BoxStyle("Round", pad=0.3),
        zorder=zorder
    )

    cos_theta = state[3]
    sin_theta = state[4]
    theta = np.arctan2(sin_theta, cos_theta)
    rotation = transforms.Affine2D().rotate_deg_around(
        state[0], state[1], np.degrees(theta) - 90
    ) + ax.transData
    rectangle.set_transform(rotation)
    ax.add_patch(rectangle)

    if plot_heading_line:
        heading_length = length / 2 + 1.5
        vehicle_center = state[:2]
        line_end_x = vehicle_center[0] + heading_length * math.cos(theta)
        line_end_y = vehicle_center[1] + heading_length * math.sin(theta)
        ax.plot(
            [vehicle_center[0], line_end_x],
            [vehicle_center[1], line_end_y],
            color='black',
            alpha=0.5,
            linewidth=heading_linewidth,
            zorder=zorder + 1
        )


def _draw_agent_goal(ax, state, color, goal_marker_size, goal_linewidth,
                     line_zorder=6, marker_zorder=7, center_zorder=8):
    """Draw a single agent's goal: a dotted line from the agent to its goal and
    an ``x`` marker at the goal, both in the agent's colour. ``state`` is the
    ``[..., goal_x, goal_y]`` (>=9 col) agent layout. When the goal coincides
    with the agent (within ``MIN_DISTANCE_TO_GOAL``) only a black ``x`` is drawn
    at the agent centre. Shared by the normal agents and the adversarial agent so
    both render their goal identically (only the colour and zorders differ)."""
    if state.shape[0] < 9:
        return
    goal = state[7:9]
    if not np.all(np.isfinite(goal)):
        return
    vehicle_center = state[:2]
    if np.linalg.norm(goal - vehicle_center) < MIN_DISTANCE_TO_GOAL:
        ax.scatter(
            vehicle_center[0], vehicle_center[1],
            marker='x', color='black', s=goal_marker_size,
            linewidths=max(goal_linewidth * 2.0, 1.2), zorder=center_zorder,
        )
        return
    ax.plot(
        [state[0], goal[0]], [state[1], goal[1]],
        color=color, linestyle=':', alpha=0.8,
        linewidth=goal_linewidth, zorder=line_zorder,
    )
    ax.scatter(
        goal[0], goal[1],
        marker='x', color=color, s=goal_marker_size,
        linewidths=max(goal_linewidth, 0.5), zorder=marker_zorder,
    )


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
    rollout convention (``ddpo.viz.CONTROL_COLOR``)."""

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

    x_range = x_max - x_min
    y_range = y_max - y_min
    scale_factor = max(x_range, y_range) / 64  # Scale based on 64m x 64m reference
    base_linewidth = 1.5 / scale_factor
    road_width = 20 / scale_factor
    scatter_size = 8 / (scale_factor ** 2)
    bbox_linewidth = 0.35 / scale_factor
    heading_linewidth = 0.3 / scale_factor
    route_linewidth = 1.5 / scale_factor
    goal_marker_size = 28 / (scale_factor ** 2)
    goal_linewidth = 0.6 / scale_factor

    ct = 0
    for i in range(len(road_points)):
        if lane_types is None: # Centerlines
            color = 'grey'
            linestyle='dashed'
            zorder = 2 
        elif lane_types[i] == 0: # Centerlines
            color = 'grey'
            linestyle='dashed'
            zorder = 2 
        elif lane_types[i] == 1: # Green traffic light lanes
            color = 'green'
            linestyle='dashed'
            zorder = 3 
        else:
            color = 'red'
            linestyle='dashed' # Red traffic light lanes
            zorder = 3
        
        lane = road_points[i, :, :2]
        ax.plot(lane[:, 0], lane[:, 1], color=color, linewidth=base_linewidth, linestyle=linestyle, zorder=zorder)
        ct += 1
        
        # Road width
        draw_road_width = False
        if lane_types is None: # only centerlines
            draw_road_width = True
        elif lane_types[i] == 0:
            draw_road_width = True
        
        if draw_road_width:
            ax.plot(lane[:, 0], lane[:, 1], color="lightgrey", linewidth=road_width, linestyle="solid", zorder=zorder-1)

        # Lane end points
        ax.scatter(lane[0, 0], lane[0, 1], color=color, s=scatter_size, zorder=zorder+1)
        ax.scatter(lane[-1, 0], lane[-1, 1], color=color, s=scatter_size, zorder=zorder+1)

        # Lane annotations (for debugging)
        # label_idx = len(lane) // 2
        # ax.annotate(i, (lane[label_idx, 0], lane[label_idx, 1]), zorder=20, fontsize=1)

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
        ax.plot(route[:, 0], route[:, 1], color='red', linestyle='solid', zorder=5, linewidth=route_linewidth)

    alpha = 1.0
    edgecolor = 'black'
    for a in range(len(agent_states)):
        if agent_types[a] == 0: # Vehicles
            color = '#de5959' if (a == 0) else '#87b3e6' # Light red for ego agent, Light blue for other vehicles
        elif agent_types[a] == 1: # Pedestrians
            color = '#bea9f5' # Light purple
        elif agent_types[a] == 2: # Immobile objects
            color = 'green' if lane_types is None else 'grey' # green for waymo dataset (cyclist), grey for nuplan (static objects)
        else:
            color = 'grey'  # Default color if agent type is unrecognized
        
        if lane_types is None:
            plot_heading_line = True # plot heading ling for vehicles, pedestrians, and cyclists
        else:
            plot_heading_line = agent_types[a] in [0, 1]  # Only plot heading line for vehicles and pedestrians, but not static objects

        # Draw bounding box and heading line
        _draw_agent_box(
            ax, agent_states[a], color, bbox_linewidth, heading_linewidth,
            plot_heading_line, alpha=alpha, edgecolor=edgecolor, zorder=4
        )

        _draw_agent_goal(ax, agent_states[a], color, goal_marker_size, goal_linewidth)

    # Draw adversarial agents in vivid green, on top of the normal agents.
    if adv_states is not None:
        for a in range(len(adv_states)):
            plot_heading_line = True if lane_types is None else (adv_types is None or adv_types[a] in [0, 1])
            _draw_agent_box(
                ax, adv_states[a], CONTROL_COLOR, bbox_linewidth, heading_linewidth,
                plot_heading_line, alpha=alpha, edgecolor=edgecolor, zorder=9
            )
            # Draw the adversary's goal on top of everything, matching the normal
            # agents' dotted-line + 'x' goal marker (in the adversary's green).
            _draw_agent_goal(
                ax, adv_states[a], CONTROL_COLOR, goal_marker_size, goal_linewidth,
                line_zorder=10, marker_zorder=11, center_zorder=11,
            )

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

    ``adv_cond`` (optional, ``[batch_size, 3]`` of discretized [type, motion,
    dist] labels) is written as a text overlay on each scene plot.
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
    clip = ImageSequenceClip(images, fps=20)
    
    # Write the video file
    clip.write_videofile(f"{image_folder}.mp4", codec='libx264')

    if delete_images:
        for image in images:
            os.remove(image)

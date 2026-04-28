"""
    Evaluation script for brick-laying RL model.
    
    This script evaluates a trained PPO policy over 100 episodes and computes:
    1. Success rate (reaching phase 3)
    2. Average placement error (distance from brick to target)
    3. Average orientation error (deviation from 90 degrees)
    
    Usage:
        python evaluate_model.py --actor_model ppo_actor.pth --target "-0.424,-4.6,0.5"
"""

import argparse
import gym
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import Mujoco_Gripper

from network import FeedForwardNN


def parse_target(target_str):
    """Parse target position string into numpy array."""
    coords = [float(x.strip()) for x in target_str.split(',')]
    if len(coords) != 3:
        raise ValueError("Target must have exactly 3 coordinates (x, y, z)")
    return np.array(coords)


def get_brick_orientation_angle(sim):
    """
    Get the orientation angle of box_1 (brick) in degrees.
    Returns the rotation angle around the vertical axis (yaw).
    """
    try:
        # Get the brick's geometry transformation matrix
        cube_geom_id = sim.model.geom_name2id('brick_geom4')
        brick_mat_3x3 = sim.data.geom_xmat[cube_geom_id].reshape(3, 3)
        #print("brick_mat_3x3:",brick_mat_3x3)
        #print('brick_geom4 orientation: ',sim.data.get_geom_xmat('brick_geom4'))
        
        startindex, endindex = sim.model.get_joint_qpos_addr('box_1_joint')
        free_joint_qpos_data = sim.data.qpos[startindex:endindex]
        free_position = free_joint_qpos_data[:3]
        free_orientation = free_joint_qpos_data[3:]
        #print(free_orientation)
        free_orientation_scipy = ([free_orientation[1],free_orientation[2],free_orientation[3],free_orientation[0]])
        r = R.from_quat(free_orientation_scipy)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        #print("roll",roll)
        #print("pitch",pitch)
        #print("yaw",yaw)

        #create a rotation object
        rotation_end = R.from_quat(free_orientation_scipy)
        rotation = np.degrees(rotation_end.magnitude())
        
        
        # Convert to rotation object and extract euler angles
        r_brick = R.from_matrix(brick_mat_3x3)
        #print("r_brick",r_brick.as_quat())
        euler_xyz = r_brick.as_euler('xyz', degrees=True)


        
        # Return the Y-axis rotation (typically the relevant rotation for brick orientation)
        # This represents the rotation around the vertical axis
        #return rotation#euler_xyz[2]  # Z-axis rotation (yaw)
        return yaw
    except Exception as e:
        print(f"Warning: Could not get brick orientation: {e}")
        return 0.0


def get_brick_position(sim):
    """Get the position of box_1 (brick)."""
    try:
        brick_pos = sim.data.get_geom_xpos('box_1')
        return brick_pos
    except Exception as e:
        print(f"Warning: Could not get brick position: {e}")
        return np.array([0.0, 0.0, 0.0])


def evaluate_episode(policy, env, target_position, max_timesteps=800, render=False):
    """
    Run a single episode and collect metrics.
    
    Returns:
        success: bool - whether phase 3 was reached
        final_brick_pos: np.array - final position of the brick
        final_orientation: float - final orientation angle of the brick (degrees)
        episode_length: int - number of timesteps in episode
        max_phase: int - maximum phase reached
    """
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle case where reset returns (obs, info)
    
    done = False
    t = 0
    max_phase = 0
    
    while not done and t < max_timesteps:
        t += 1
        
        if render:
            env.render()
        
        # Query deterministic action from policy
        action = policy(obs).detach().numpy()
        obs, rew, done, info = env.step(action)
        
        # Track maximum phase reached
        current_phase = env.phase
        max_phase = max(max_phase, current_phase)
    
    # Get final brick state
    final_brick_pos = get_brick_position(env.sim)
    #print(final_brick_pos)
    final_orientation = get_brick_orientation_angle(env.sim)
    
    success = (max_phase >= 3)
    
    return success, final_brick_pos, final_orientation, t, max_phase


def compute_placement_error(brick_pos, target_pos):
    """Compute Euclidean distance between brick and target position."""
    return np.linalg.norm(brick_pos - target_pos)


def compute_orientation_error(orientation_angle, target_angle=90.0):
    """
    Compute orientation error relative to target angle (default 90 degrees).
    Handles wrap-around for angles.
    """
    diff = abs(orientation_angle - target_angle)
    if diff > 90:
    	diff = abs ( diff - 180)	
    # Handle angle wrap-around (consider both 90 and -90 as equivalent for some cases)
    #diff = min(diff, abs(diff - 180), abs(diff + 180))
    return diff


def main():
    parser = argparse.ArgumentParser(description='Evaluate brick-laying RL model')
    parser.add_argument('--actor_model', type=str, required=True,
                        help='Path to the trained actor model (.pth file)')
    parser.add_argument('--target', type=str, default="-0.424,-4.6,0.5",
                        help='Target position as "x,y,z" (default: "-0.424,-4.6,0.5")')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to evaluate (default: 100)')
    parser.add_argument('--max_timesteps', type=int, default=800,
                        help='Maximum timesteps per episode (default: 800)')
    parser.add_argument('--target_orientation', type=float, default=90.0,
                        help='Target orientation angle in degrees (default: 90)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Print details for each episode')
    
    args = parser.parse_args()
    
    # Parse target position
    target_position = parse_target(args.target)
    print(f"\n{'='*60}")
    print(f"BRICK-LAYING MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Actor model: {args.actor_model}")
    print(f"Target position: {target_position}")
    print(f"Target orientation: {args.target_orientation}°")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Max timesteps per episode: {args.max_timesteps}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = gym.make('BrickLaying-v1')
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    
    # Build policy network and load weights
    policy = FeedForwardNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(args.actor_model, map_location='cpu'))
    policy.eval()  # Set to evaluation mode
    
    print(f"Successfully loaded model: {args.actor_model}\n")
    
    # Storage for metrics
    successes = []
    placement_errors = []
    orientation_errors = []
    episode_lengths = []
    max_phases = []
    
    # Successful episode metrics (only for episodes that reached phase 3)
    successful_placement_errors = []
    successful_orientation_errors = []
    
    print(f"Running {args.num_episodes} episodes...")
    print("-" * 60)
    
    for ep in range(args.num_episodes):
        success, final_pos, final_orientation, ep_length, max_phase = evaluate_episode(
            policy, env, target_position, 
            max_timesteps=args.max_timesteps,
            render=args.render
        )
        
        # Compute errors
        placement_error = compute_placement_error(final_pos, target_position)
        orientation_error = compute_orientation_error(final_orientation, args.target_orientation)
        
        # Store metrics
        successes.append(success)
        placement_errors.append(placement_error)
        orientation_errors.append(orientation_error)
        episode_lengths.append(ep_length)
        max_phases.append(max_phase)
        
        if success:
            successful_placement_errors.append(placement_error)
            successful_orientation_errors.append(orientation_error)
        
        if args.verbose or (ep + 1) % 10 == 0:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"Episode {ep + 1:3d}/{args.num_episodes}: {status} | "
                  f"Phase: {max_phase} | "
                  f"Length: {ep_length:4d} | "
                  f"Placement Error: {placement_error:.4f}m | "
                  f"Orientation Error: {orientation_error:.2f}°")
    
    # Compute summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    total_success = sum(successes)
    success_rate = (total_success / args.num_episodes) * 100
    
    print(f"\n📊 SUCCESS METRICS:")
    print(f"   Successful episodes (Phase 3 reached): {total_success}/{args.num_episodes}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    print(f"\n📍 PLACEMENT ERROR (all episodes):")
    print(f"   Average: {np.mean(placement_errors):.4f} m")
    print(f"   Std Dev: {np.std(placement_errors):.4f} m")
    print(f"   Min:     {np.min(placement_errors):.4f} m")
    print(f"   Max:     {np.max(placement_errors):.4f} m")
    
    print(f"\n🔄 ORIENTATION ERROR (all episodes):")
    print(f"   Target orientation: {args.target_orientation}°")
    print(f"   Average: {np.mean(orientation_errors):.2f}°")
    print(f"   Std Dev: {np.std(orientation_errors):.2f}°")
    print(f"   Min:     {np.min(orientation_errors):.2f}°")
    print(f"   Max:     {np.max(orientation_errors):.2f}°")
    
    if successful_placement_errors:
        print(f"\n📍 PLACEMENT ERROR (successful episodes only):")
        print(f"   Average: {np.mean(successful_placement_errors):.4f} m")
        print(f"   Std Dev: {np.std(successful_placement_errors):.4f} m")
        print(f"   Min:     {np.min(successful_placement_errors):.4f} m")
        print(f"   Max:     {np.max(successful_placement_errors):.4f} m")
        
        print(f"\n🔄 ORIENTATION ERROR (successful episodes only):")
        print(f"   Average: {np.mean(successful_orientation_errors):.2f}°")
        print(f"   Std Dev: {np.std(successful_orientation_errors):.2f}°")
        print(f"   Min:     {np.min(successful_orientation_errors):.2f}°")
        print(f"   Max:     {np.max(successful_orientation_errors):.2f}°")
    
    print(f"\n⏱️ EPISODE LENGTH:")
    print(f"   Average: {np.mean(episode_lengths):.1f} steps")
    print(f"   Std Dev: {np.std(episode_lengths):.1f} steps")
    
    # Phase distribution
    print(f"\n📈 PHASE DISTRIBUTION:")
    for phase in range(4):
        count = sum(1 for p in max_phases if p == phase)
        pct = (count / args.num_episodes) * 100
        print(f"   Phase {phase}: {count} episodes ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    
    # Close environment
    env.close()
    
    # Return summary dict for programmatic use
    return {
        'success_rate': success_rate,
        'avg_placement_error': np.mean(placement_errors),
        'avg_orientation_error': np.mean(orientation_errors),
        'successful_placement_error': np.mean(successful_placement_errors) if successful_placement_errors else None,
        'successful_orientation_error': np.mean(successful_orientation_errors) if successful_orientation_errors else None,
        'total_successes': total_success,
        'total_episodes': args.num_episodes
    }


if __name__ == '__main__':
    main()

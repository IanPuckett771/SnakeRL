"""
Train multiple RL algorithms and compare them on wandb.
"""
import time
import json
import argparse
import numpy as np
from pathlib import Path

import wandb

from game.engine import SnakeGame
from algorithms import DQNAgent, DQNCNNAgent, PPOAgent, A2CAgent


NUM_PARALLEL_GAMES = 8  # Run this many games simultaneously for faster training
NUM_PARALLEL_GAMES_CNN = 2  # Fewer for CNN (more compute per step)


def train_algorithm(agent, algorithm_name, duration_seconds=60, use_wandb=True, 
                   project_name="snakerl-comparison", run_id=None):
    """Train a single algorithm with parallel game environments."""
    # Use fewer parallel envs for CNN (more compute per step)
    is_cnn = hasattr(agent, 'board_width')  # CNN agent has board_width attribute
    num_envs = NUM_PARALLEL_GAMES_CNN if is_cnn else NUM_PARALLEL_GAMES
    
    print(f"\n{'='*60}")
    print(f"Training {algorithm_name} ({num_envs} parallel envs{' [CNN]' if is_cnn else ''})")
    print(f"{'='*60}\n")
    
    # Generate unique run ID if not provided
    if run_id is None:
        run_id = int(time.time())
    
    # Initialize wandb run with timeout to prevent hanging
    run = None
    if use_wandb:
        try:
            import os
            os.environ["WANDB_INIT_TIMEOUT"] = "30"  # 30 second timeout
            os.environ["WANDB_START_METHOD"] = "thread"
            run = wandb.init(
                project=project_name,
                name=f"{algorithm_name}-{int(time.time())}",
                group="algorithm-comparison",
                tags=[algorithm_name, "comparison"],
                config={
                    "algorithm": algorithm_name,
                    "duration_seconds": duration_seconds,
                    "parallel_envs": num_envs,
                },
                reinit=True,
                settings=wandb.Settings(init_timeout=30)
            )
            print(f"[OK] Wandb initialized: {run.url}")
            print(f"   View at: https://wandb.ai/{run.entity}/{project_name}/groups/algorithm-comparison")
        except Exception as e:
            print(f"[WARNING] Failed to initialize wandb: {e}")
            print("   Continuing without wandb logging...")
            use_wandb = False
    
    # Create training lock file
    training_lock_file = Path(".training_lock")
    start_time = time.time()
    
    try:
        with open(training_lock_file, 'w') as f:
            json.dump({
                "start_time": start_time,
                "duration": duration_seconds,
                "episodes": 0,
                "avg_score": 0,
                "algorithm": algorithm_name,
            }, f)
        
        # Create parallel game environments
        games = [SnakeGame(width=20, height=20) for _ in range(num_envs)]
        states = [g.reset() for g in games]
        game_rewards = [0.0] * num_envs
        game_steps = [0] * num_envs
        
        episode = 0
        scores = []
        episode_rewards = []
        episode_lengths = []
        snake_lengths = []
        losses = []
        
        # Setup for intermediate checkpoints (10 stages)
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        num_stages = 10
        checkpoint_interval = duration_seconds / num_stages
        last_checkpoint_time = start_time
        stage = 0
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use the agent's algorithm_name for prefix (e.g. "dqn" or "dqn-cnn")
        prefix_name = agent.algorithm_name.lower().replace("-", "_")
        checkpoint_prefix = f"{prefix_name}_agent_{timestamp}"
        print(f"Checkpoint prefix: {checkpoint_prefix}")
        print(f"Checkpoints will be saved as: {checkpoint_prefix}_stage##.pt\n")
        
        total_steps = 0
        log_interval = 10  # Log every N completed episodes
        last_logged_episode = 0  # Track to avoid duplicate logging
        
        # Training loop - step all games each iteration
        while time.time() - start_time < duration_seconds:
            # Step all parallel games
            for i in range(num_envs):
                if games[i].game_over:
                    # Episode finished - record stats and reset
                    episode += 1
                    agent.episode = episode
                    snake_length = len(games[i].snake)
                    
                    scores.append(games[i].score)
                    episode_rewards.append(game_rewards[i])
                    episode_lengths.append(game_steps[i])
                    snake_lengths.append(snake_length)
                    
                    # Reset this game
                    states[i] = games[i].reset()
                    game_rewards[i] = 0.0
                    game_steps[i] = 0
                    continue
                
                old_state = states[i]
                action = agent.get_action(old_state, training=True)
                
                new_state, reward, done = games[i].step(action)
                game_rewards[i] += reward
                game_steps[i] += 1
                total_steps += 1
                
                if algorithm_name == "DQN":
                    agent.remember(old_state, action, reward, new_state, done)
                    if len(agent.memory) > agent.batch_size:
                        loss = agent.update()
                        if loss > 0:
                            losses.append(loss)
                else:
                    agent.store_reward(reward, done)
                    if algorithm_name == "A2C" and (done or game_steps[i] >= agent.n_steps):
                        loss = agent.update()
                        if loss > 0:
                            losses.append(loss)
                    if algorithm_name == "PPO" and done:
                        loss = agent.update()
                        if loss > 0:
                            losses.append(loss)
                
                states[i] = new_state
                
                # Prevent infinite loops per game - scale with snake length
                # Short snake: 1000 steps max. Long snake needs much more time.
                snake_len = len(games[i].snake)
                max_steps = max(1000, snake_len * 10, 5000)  # At least 5000 for long snakes
                if game_steps[i] > max_steps:
                    games[i].game_over = True
            
            # Log metrics periodically (only when new episodes have completed)
            if episode > 0 and episode % log_interval == 0 and episode > last_logged_episode and len(scores) >= log_interval:
                last_logged_episode = episode
                avg_score = np.mean(scores[-log_interval:])
                avg_reward = np.mean(episode_rewards[-log_interval:])
                avg_length = np.mean(episode_lengths[-log_interval:])
                avg_snake_length = np.mean(snake_lengths[-log_interval:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                eps_per_sec = episode / max(elapsed, 1)
                
                # Update training lock file
                try:
                    with open(training_lock_file, 'w') as f:
                        json.dump({
                            "start_time": start_time,
                            "duration": duration_seconds,
                            "episodes": episode,
                            "avg_score": float(avg_score),
                            "avg_snake_length": float(avg_snake_length),
                            "algorithm": algorithm_name,
                            "eps_per_sec": round(eps_per_sec, 1),
                        }, f)
                except:
                    pass
                
                # Save intermediate checkpoint if enough time has passed
                elapsed_time = time.time() - start_time
                time_since_last_checkpoint = elapsed_time - (last_checkpoint_time - start_time)
                if time_since_last_checkpoint >= checkpoint_interval and stage < num_stages:
                    stage += 1
                    checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_stage{stage:02d}.pt"
                    agent.save_checkpoint(str(checkpoint_path))
                    last_checkpoint_time = time.time()
                    print(f"  -> Saved checkpoint stage {stage}/{num_stages}: {checkpoint_path.name} (Avg Score: {avg_score:.2f})")
                
                # Log to wandb
                if use_wandb:
                    log_dict = {
                        "episode": episode,
                        "score": scores[-1] if scores else 0,
                        "avg_score_10": avg_score,
                        "avg_score_all": np.mean(scores) if scores else 0,
                        "episode_reward": episode_rewards[-1] if episode_rewards else 0,
                        "avg_reward_10": avg_reward,
                        "episode_length": episode_lengths[-1] if episode_lengths else 0,
                        "avg_length_10": avg_length,
                        "snake_length": snake_lengths[-1] if snake_lengths else 0,
                        "avg_snake_length_10": avg_snake_length,
                        "avg_snake_length_all": np.mean(snake_lengths) if snake_lengths else 0,
                        "time_elapsed": elapsed,
                        "time_remaining": remaining,
                        "stage": stage,
                        "total_steps": total_steps,
                        "eps_per_sec": eps_per_sec,
                    }
                    if avg_loss > 0:
                        log_dict["loss"] = avg_loss
                    if algorithm_name == "DQN":
                        log_dict["epsilon"] = agent.epsilon
                    
                    wandb.log(log_dict)
                
                print(f"Ep {episode:5d} | Score: {scores[-1]:3d} | "
                      f"Avg(10): {avg_score:6.2f} | "
                      f"SnakeLen: {avg_snake_length:5.1f} | "
                      f"Eps/s: {eps_per_sec:5.1f} | "
                      f"Steps: {total_steps:>8d} | "
                      f"Left: {remaining:5.0f}s | "
                      f"Stage: {stage}/{num_stages}")
        
        # Save final checkpoint (stage 10 or final)
        final_checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_stage{num_stages:02d}.pt"
        agent.save_checkpoint(str(final_checkpoint_path))
        # Also save as the main checkpoint for backward compatibility (overwrites previous)
        main_checkpoint_path = checkpoint_dir / f"{prefix_name}_agent.pt"
        agent.save_checkpoint(str(main_checkpoint_path))
        
        # Final statistics
        final_avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores) if scores else 0
        final_avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards) if episode_rewards else 0
        
        print(f"\n{'-'*60}")
        print(f"Training complete for {algorithm_name}!")
        print(f"Episodes: {episode}")
        print(f"Final average score: {final_avg_score:.2f}")
        print(f"Final average reward: {final_avg_reward:.2f}")
        print(f"Checkpoints saved:")
        print(f"  - Final: {final_checkpoint_path.name}")
        print(f"  - Main: {main_checkpoint_path.name}")
        print(f"  - Total stages: {stage}/{num_stages}")
        print(f"{'-'*60}\n")
        
        # Log final metrics
        if use_wandb and run:
            wandb.log({
                "final_avg_score": final_avg_score,
                "final_avg_reward": final_avg_reward,
                "total_episodes": episode,
            })
            wandb.finish()
            print(f"[OK] Wandb run completed: {run.url}")
        
        return {
            "algorithm": algorithm_name,
            "episodes": episode,
            "final_avg_score": final_avg_score,
            "final_avg_reward": final_avg_reward,
            "checkpoint": str(main_checkpoint_path),
            "stages": stage,
        }
        
    finally:
        # Always remove training lock file
        if training_lock_file.exists():
            try:
                training_lock_file.unlink()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Train and compare multiple RL algorithms")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Training duration per algorithm in seconds (default: 60)")
    parser.add_argument("--no-wandb", action="store_true", 
                       help="Disable wandb logging")
    parser.add_argument("--project", type=str, default="snakerl-comparison",
                       help="Wandb project name (default: snakerl-comparison)")
    parser.add_argument("--algorithms", nargs="+", 
                       choices=["DQN", "PPO", "A2C", "all"],
                       default=["all"],
                       help="Which algorithms to train (default: all)")
    parser.add_argument("--fresh", action="store_true",
                       help="Start fresh instead of resuming from previous checkpoint")
    parser.add_argument("--cnn", action="store_true",
                       help="Use CNN-based DQN (sees entire board) instead of flat features")
    
    args = parser.parse_args()
    
    # Determine which algorithms to train
    if "all" in args.algorithms:
        algorithms_to_train = ["DQN", "PPO", "A2C"]
    else:
        algorithms_to_train = args.algorithms
    
    model_type = "CNN" if args.cnn else "Flat"
    
    print(f"\n{'='*60}")
    print(f"SnakeRL Algorithm Comparison")
    print(f"{'='*60}")
    print(f"Algorithms: {', '.join(algorithms_to_train)}")
    print(f"Model type: {model_type}")
    print(f"Duration per algorithm: {args.duration} seconds")
    print(f"Wandb: {'Disabled' if args.no_wandb else f'Enabled (project: {args.project})'}")
    print(f"Resume from checkpoint: {'No (fresh start)' if args.fresh else 'Yes (if available)'}")
    print(f"{'='*60}\n")
    
    results = []
    checkpoint_dir = Path("checkpoints")
    
    # Train each algorithm
    for alg_name in algorithms_to_train:
        # Create agent with improved settings for better learning
        if alg_name == "DQN":
            if args.cnn:
                # CNN-based DQN — sees the entire board as a 7-channel grid
                agent = DQNCNNAgent(
                    lr=0.0005,
                    gamma=0.99,
                    epsilon=1.0,
                    epsilon_min=0.05,
                    epsilon_decay=0.9995,
                    memory_size=50000,  # Smaller than flat (grids use more memory)
                    batch_size=64,      # Smaller batches (CNN is more compute-heavy)
                )
                checkpoint_name = "dqn_cnn_agent.pt"
                print(f"[CNN] Using DQN-CNN agent (7-channel grid input, full board vision)")
            else:
                # Flat feature vector DQN
                agent = DQNAgent(
                    lr=0.0005,
                    gamma=0.99,
                    epsilon=1.0,
                    epsilon_min=0.05,
                    epsilon_decay=0.9995,
                    memory_size=100000,
                    batch_size=256,
                )
                checkpoint_name = "dqn_agent.pt"
            
            # Try to load from previous checkpoint unless --fresh is specified
            if not args.fresh:
                main_checkpoint = checkpoint_dir / checkpoint_name
                if main_checkpoint.exists():
                    print(f"[RESUME] Found previous checkpoint: {main_checkpoint.name}")
                    agent.load_checkpoint(str(main_checkpoint))
                else:
                    print(f"[NEW] No previous checkpoint found, starting fresh")
                    
        elif alg_name == "PPO":
            agent = PPOAgent()
        elif alg_name == "A2C":
            agent = A2CAgent()
        else:
            continue
        
        # Train
        result = train_algorithm(
            agent, 
            alg_name, 
            duration_seconds=args.duration,
            use_wandb=not args.no_wandb,
            project_name=args.project
        )
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Algorithm':<10} {'Episodes':<10} {'Avg Score':<12} {'Avg Reward':<12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['algorithm']:<10} {r['episodes']:<10} {r['final_avg_score']:<12.2f} {r['final_avg_reward']:<12.2f}")
    print(f"{'='*60}\n")
    
    # Find best algorithm
    if results:
        best = max(results, key=lambda x: x['final_avg_score'])
        print(f"[BEST] Algorithm: {best['algorithm']} (Score: {best['final_avg_score']:.2f})")
        print(f"   Checkpoint: {best['checkpoint']}\n")


if __name__ == "__main__":
    main()

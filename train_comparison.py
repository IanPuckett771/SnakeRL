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
from algorithms import DQNAgent, PPOAgent, A2CAgent


def train_algorithm(agent, algorithm_name, duration_seconds=60, use_wandb=True, 
                   project_name="snakerl-comparison", run_id=None):
    """Train a single algorithm."""
    print(f"\n{'='*60}")
    print(f"Training {algorithm_name}")
    print(f"{'='*60}\n")
    
    # Generate unique run ID if not provided
    if run_id is None:
        run_id = int(time.time())
    
    # Initialize wandb run
    run = None
    if use_wandb:
        try:
            run = wandb.init(
                project=project_name,
                name=f"{algorithm_name}-{int(time.time())}",
                group="algorithm-comparison",
                tags=[algorithm_name, "comparison"],
                config={
                    "algorithm": algorithm_name,
                    "duration_seconds": duration_seconds,
                },
                reinit=True
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
        
        game = SnakeGame(width=20, height=20)
        episode = 0
        scores = []
        episode_rewards = []
        episode_lengths = []
        snake_lengths = []  # Track snake length for each episode
        losses = []
        
        # Setup for intermediate checkpoints (10 stages)
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        num_stages = 10
        checkpoint_interval = duration_seconds / num_stages
        last_checkpoint_time = start_time
        stage = 0
        
        # Use run_id to create unique checkpoint names with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_prefix = f"{algorithm_name.lower()}_agent_{timestamp}"
        print(f"Checkpoint prefix: {checkpoint_prefix}")
        print(f"Checkpoints will be saved as: {checkpoint_prefix}_stage##.pt\n")
        
        # Training loop
        while time.time() - start_time < duration_seconds:
            episode += 1
            agent.episode = episode
            state = game.reset()
            total_reward = 0
            steps = 0
            episode_loss = []
            
            while not game.game_over:
                # Get action from current state
                old_state = state
                action = agent.get_action(state, training=True)
                
                # Take step
                state, reward, done = game.step(action)
                total_reward += reward
                steps += 1
                
                if algorithm_name == "DQN":
                    # Store experience for DQN (old_state -> action -> reward -> new_state)
                    agent.remember(old_state, action, reward, state, done)
                    
                    # Update periodically
                    if len(agent.memory) > agent.batch_size:
                        loss = agent.update()
                        if loss > 0:
                            episode_loss.append(loss)
                else:
                    # PPO or A2C - store reward
                    agent.store_reward(reward, done)
                    
                    # Update for A2C every n_steps or on done
                    if algorithm_name == "A2C" and (done or steps >= agent.n_steps):
                        loss = agent.update()
                        if loss > 0:
                            episode_loss.append(loss)
                
                # Prevent infinite loops
                if steps > 1000:
                    break
            
            # Update for PPO at end of episode
            if algorithm_name == "PPO" and game.game_over:
                loss = agent.update()
                if loss > 0:
                    episode_loss.append(loss)
            
            # Track snake length at end of episode
            snake_length = len(game.snake)
            
            scores.append(game.score)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            snake_lengths.append(snake_length)
            if episode_loss:
                losses.append(np.mean(episode_loss))
            
            # Log metrics every 10 episodes
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:]) if scores else 0
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
                avg_snake_length = np.mean(snake_lengths[-10:]) if snake_lengths else 0
                avg_loss = np.mean(losses[-10:]) if losses else 0
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                
                # Update training lock file
                try:
                    with open(training_lock_file, 'w') as f:
                        json.dump({
                            "start_time": start_time,
                            "duration": duration_seconds,
                            "episodes": episode,
                            "avg_score": avg_score,
                            "algorithm": algorithm_name,
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
                        "score": game.score,
                        "avg_score_10": avg_score,
                        "avg_score_all": np.mean(scores) if scores else 0,
                        "episode_reward": total_reward,
                        "avg_reward_10": avg_reward,
                        "episode_length": steps,
                        "avg_length_10": avg_length,
                        "snake_length": snake_length,
                        "avg_snake_length_10": avg_snake_length,
                        "avg_snake_length_all": np.mean(snake_lengths) if snake_lengths else 0,
                        "time_elapsed": elapsed,
                        "time_remaining": remaining,
                        "stage": stage,
                    }
                    if avg_loss > 0:
                        log_dict["loss"] = avg_loss
                    if algorithm_name == "DQN":
                        log_dict["epsilon"] = agent.epsilon
                    
                    wandb.log(log_dict)
                
                print(f"Episode {episode:4d} | Score: {game.score:3d} | "
                      f"Avg Score (10): {avg_score:6.2f} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Steps: {steps:4d} | "
                      f"Time left: {remaining:5.1f}s | "
                      f"Stage: {stage}/{num_stages}")
        
        # Save final checkpoint (stage 10 or final)
        final_checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_stage{num_stages:02d}.pt"
        agent.save_checkpoint(str(final_checkpoint_path))
        # Also save as the main checkpoint for backward compatibility (overwrites previous)
        main_checkpoint_path = checkpoint_dir / f"{algorithm_name.lower()}_agent.pt"
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
    
    args = parser.parse_args()
    
    # Determine which algorithms to train
    if "all" in args.algorithms:
        algorithms_to_train = ["DQN", "PPO", "A2C"]
    else:
        algorithms_to_train = args.algorithms
    
    print(f"\n{'='*60}")
    print(f"SnakeRL Algorithm Comparison")
    print(f"{'='*60}")
    print(f"Algorithms: {', '.join(algorithms_to_train)}")
    print(f"Duration per algorithm: {args.duration} seconds")
    print(f"Wandb: {'Disabled' if args.no_wandb else f'Enabled (project: {args.project})'}")
    print(f"{'='*60}\n")
    
    results = []
    
    # Train each algorithm
    for alg_name in algorithms_to_train:
        # Create agent with improved settings for better learning
        if alg_name == "DQN":
            agent = DQNAgent(
                lr=0.001,  # Learning rate
                gamma=0.99,  # Higher discount for long-term planning
                epsilon=1.0,  # Start with full exploration
                epsilon_min=0.05,  # Keep some exploration
                epsilon_decay=0.9995,  # Slow decay for 30 min training
                memory_size=50000,  # Larger memory for more experience
                batch_size=128  # Larger batches for more stable learning
            )
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

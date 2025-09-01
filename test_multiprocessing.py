import sys
from train import AlphaZeroTrainer

if __name__ == "__main__":
    # Create a minimal AlphaZeroTrainer with multiprocessing
    trainer = AlphaZeroTrainer(
        game_size=6,  # Smaller board for faster testing
        num_iterations=1,
        num_self_play_games=2,  # Just 2 games for quick test
        num_mcts_simulations=10,  # Low number for quick test
        use_multiprocessing=True,
        mp_num_workers=2,
        mp_games_per_worker=1
    )
    
    # Test only the multiprocessing self-play function
    try:
        print("Testing multiprocessing self-play...")
        examples = trainer.self_play_multiprocess(num_workers=2, games_per_worker=1)
        print(f"Successfully generated {len(examples)} examples")
        print("Test passed!")
        sys.exit(0)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
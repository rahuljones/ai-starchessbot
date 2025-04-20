import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def play_single_match(bot1, bot2, game_index):
    result = subprocess.run(
        ["python", "versus.py", "--bot1", bot1, "--bot2", bot2],
        capture_output=True,
        text=True
    )
    output = result.stdout

    if "White wins!" in output:
        return game_index, "BOT1"
    elif "Black wins!" in output:
        return game_index, "BOT2"
    else:
        return game_index, "DRAW"

def run_matches(bot1, bot2, num_matches=1000, max_workers=20):
    bot1_wins = 0
    bot2_wins = 0
    draws = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(play_single_match, bot1, bot2, i): i for i in range(num_matches)}

        for future in as_completed(futures):
            game_index, result = future.result()
            if result == "BOT1":
                bot1_wins += 1
                print(f"Game {game_index}, Result: BOT1: {bot1} wins")
            elif result == "BOT2":
                bot2_wins += 1
                print(f"Game {game_index}, Result: BOT2: {bot2} wins")
            else:
                draws += 1
                print(f"Game {game_index}, Result: Draw")

    bot1_win_percentage = (bot1_wins / num_matches) * 100
    bot2_win_percentage = (bot2_wins / num_matches) * 100

    print(f"{bot1} Win Percentage: {bot1_win_percentage:.2f}%")
    print(f"{bot2} Win Percentage: {bot2_win_percentage:.2f}%")
    print(f"Draw Percentage: {(draws / num_matches) * 100:.2f}%")

if __name__ == "__main__":
    BOT1 = "God2Bot"
    BOT2 = "God1Bot"
    run_matches(BOT1, BOT2)
    print("switching...")
    run_matches(BOT2, BOT1)
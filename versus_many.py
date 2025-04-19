import subprocess

def run_matches(bot1, bot2, num_matches=10):
    bot1_wins = 0
    bot2_wins = 0

    for i in range(num_matches):
        result = subprocess.run(
            ["python", "versus.py", "--bot1", bot1, "--bot2", bot2],
            capture_output=True,
            text=True
        )
        output = result.stdout

        if "White wins!" in output:
            bot1_wins += 1
            print(f"Game {i}, Result: BOT1: {bot1} wins")

        elif "Black wins!" in output:
            bot2_wins += 1
            print(f"Game {i}, Result: BOT2: {bot2} wins")
        else: 
            print(f"Game {i}, Result: Draw")
        

    bot1_win_percentage = (bot1_wins / num_matches) * 100
    bot2_win_percentage = (bot2_wins / num_matches) * 100

    print(f"{bot1} Win Percentage: {bot1_win_percentage:.2f}%")
    print(f"{bot2} Win Percentage: {bot2_win_percentage:.2f}%")

if __name__ == "__main__":
    # parser.add_argument("--bot1", type=str, default="IterativeBot", help="Choose bot1 (e.g., RandomBot, SingleStepBot, MinimaxBot, MultiThreadedMinimaxBot, MonteCarloBot, IterativeBot)")
    BOT1 = "IterativeBotH"
    BOT2 = "IterativeBot"
    run_matches(BOT1, BOT2)
    print("switching...")
    run_matches(BOT2, BOT1)
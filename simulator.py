import pygame
import argparse
import importlib

from data.classes.Board import Board

pygame.init()


WINDOW_SIZE = (600, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)


def draw(display, board):
    display.fill("white")
    board.draw(display)
    pygame.display.update()


def run_game(bot1_class, bot2_class, delay):
    board = Board(WINDOW_SIZE[0], WINDOW_SIZE[1])
    bot1 = bot1_class()
    bot2 = bot2_class()
    running = True

    while running:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                board.handle_click(mx, my)
                print(board.last_captured)

        if board.turn == "black":
            move = bot1.move("black", board)
        else:
            move = bot2.move("white", board)

        board.handle_move(*move)
        pygame.time.delay(delay)

        if board.is_in_checkmate("black"):
            print("White wins!")
            running = False
        elif board.is_in_checkmate("white"):
            print("Black wins!")
            running = False
        elif board.is_in_draw():
            print("Draw!")
            running = False

        draw(screen, board)


# ... [rest of the code remains unchanged above] ...

if __name__ == "__main__":
    print("Made it this far 2")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot1", type=str, default="random_bot", help="Bot for black (e.g. 'random_bot')")
    parser.add_argument("--bot2", type=str, default="random_bot", help="Bot for white (e.g. 'random_bot')")
    parser.add_argument("--delay", type=int, default=0, help="Delay in ms between moves")
    parser.add_argument("--simulations", type=int, default=1, help="Number of simulations to run")
    args = parser.parse_args()

    try:
        bot1_module = importlib.import_module(f"data.classes.bots.{args.bot1}")
        bot2_module = importlib.import_module(f"data.classes.bots.{args.bot2}")
        bot1_class = bot1_module.Bot
        bot2_class = bot2_module.Bot
    except ModuleNotFoundError as e:
        print(f"Error: Could not find bot module - {e}")
        exit()

    bot1_wins = 0
    bot2_wins = 0
    draws = 0

    for i in range(args.simulations):
        print(f"\n--- Starting Simulation {i + 1} ---")

        def run_game_with_result():
            board = Board(WINDOW_SIZE[0], WINDOW_SIZE[1])
            bot1 = bot1_class()
            bot2 = bot2_class()
            running = True

            while running:
                mx, my = pygame.mouse.get_pos()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        board.handle_click(mx, my)
                        print(board.last_captured)

                if board.turn == "black":
                    move = bot1.move("black", board)
                else:
                    move = bot2.move("white", board)

                board.handle_move(*move)
                pygame.time.delay(args.delay)
                draw(screen, board)

                if board.is_in_checkmate("black"):
                    print("White wins!")
                    return "white"
                elif board.is_in_checkmate("white"):
                    print("Black wins!")
                    return "black"
                elif board.is_in_draw():
                    print("Draw!")
                    return "draw"

        result = run_game_with_result()

        if result == "black":
            bot1_wins += 1  # bot1 always plays black
        elif result == "white":
            bot2_wins += 1  # bot2 always plays white
        elif result == "draw":
            draws += 1

    total = bot1_wins + bot2_wins + draws
    print(f"\n=== Simulation Results: {args.bot1} (Black) vs {args.bot2} (White) ===")
    print(f"{args.bot1} (bot1/black) wins: {bot1_wins} ({(bot1_wins / total) * 100:.1f}%)")
    print(f"{args.bot2} (bot2/white) wins: {bot2_wins} ({(bot2_wins / total) * 100:.1f}%)")
    print(f"Draws: {draws} ({(draws / total) * 100:.1f}%)")

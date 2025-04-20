# bot.py - Single file submission (Refined - Single-Threaded)
# Implements Iterative Deepening Minimax with Alpha-Beta, Quiescence Search,
# MVV-LVA Ordering, Transposition Table, and Internal Board Representation
# with Make/Unmake moves to avoid deep copying.

import time
import random
import copy
import math
import os
# import concurrent.futures # Removed
from collections import namedtuple

# --- Constants ---
EMPTY=0; wP,wN,wB,wR,wS,wQ,wK,wJ=1,2,3,4,5,6,7,8; bP,bN,bB,bR,bS,bQ,bK,bJ=-1,-2,-3,-4,-5,-6,-7,-8
PIECE_COLORS={p:1 for p in [wP,wN,wB,wR,wS,wQ,wK,wJ]}; PIECE_COLORS.update({p:-1 for p in [bP,bN,bB,bR,bS,bQ,bK,bJ]}); PIECE_COLORS[EMPTY]=0
EXTERNAL_TO_INTERNAL={'wP':wP,'wN':wN,'wB':wB,'wR':wR,'wS':wS,'wQ':wQ,'wK':wK,'wJ':wJ,'bP':bP,'bN':bN,'bB':bB,'bR':bR,'bS':bS,'bQ':bQ,'bK':bK,'bJ':bJ,'':EMPTY}
INTERNAL_TO_NOTATION={wP:' ',wN:'N',wB:'B',wR:'R',wS:'S',wQ:'Q',wK:'K',wJ:'J',bP:' ',bN:'N',bB:'B',bR:'R',bS:'S',bQ:'Q',bK:'K',bJ:'J'}
SCORES_DICT={wP:100,bP:-100,wN:320,bN:-320,wB:330,bB:-330,wR:500,bR:-500,wS:500,bS:-500,wQ:900,bQ:-900,wJ:900,bJ:-900,wK:20000,bK:-20000,EMPTY:0}
PAWN_TABLE=[[0,0,0,0,0,0],[10,10,0,0,10,10],[5,5,10,10,5,5],[0,0,20,20,0,0],[5,10,25,25,10,5],[0,0,0,0,0,0]]
KNIGHT_TABLE=[[-50,-40,-30,-30,-40,-50],[-40,-20,0,0,-20,-40],[-30,0,10,10,0,-30],[-30,5,15,15,5,-30],[-40,-20,0,0,-20,-40],[-50,-40,-30,-30,-40,-50]]
BISHOP_TABLE=[[-20,-10,-10,-10,-10,-20],[-10,0,0,0,0,-10],[-10,0,5,5,0,-10],[-10,5,5,5,5,-10],[-10,0,5,5,0,-10],[-20,-10,-10,-10,-10,-20]]
ROOK_TABLE=[[0,0,0,0,0,0],[5,10,10,10,10,5],[-5,0,0,0,0,-5],[-5,0,0,0,0,-5],[-5,0,0,0,0,-5],[0,0,0,5,5,0]]
QUEEN_TABLE=[[-20,-10,-10,-5,-5,-10,-20],[-10,0,0,0,0,0,-10],[-10,0,5,5,5,5,-10],[-5,0,5,5,5,5,-5],[-10,5,5,5,5,0,-10],[-20,-10,-10,-5,-5,-10,-20]]
KING_TABLE_EARLY=[[-30,-40,-40,-50,-50,-40,-30],[-30,-40,-40,-50,-50,-40,-30],[-30,-40,-40,-50,-50,-40,-30],[-30,-40,-40,-50,-50,-40,-30],[-10,-20,-20,-20,-20,-20,-10],[20,30,10,0,0,10,30,20]]
STAR_TABLE=[[-40,-30,-20,-20,-30,-40],[-30,-10,5,5,-10,-30],[-20,5,15,15,5,-20],[-20,10,15,15,10,-20],[-30,-10,5,5,-10,-30],[-40,-30,-20,-20,-30,-40]]
JOKER_TABLE=QUEEN_TABLE
PIECE_TABLES={wP:PAWN_TABLE,wN:KNIGHT_TABLE,wB:BISHOP_TABLE,wR:ROOK_TABLE,wQ:QUEEN_TABLE,wK:KING_TABLE_EARLY,wS:STAR_TABLE,wJ:JOKER_TABLE}
TT_EXACT,TT_LOWERBOUND,TT_UPPERBOUND=0,1,2
MAX_QUIESCENCE_DEPTH=5; BOARD_SIZE=6; FIFTY_MOVE_LIMIT=100
UndoInfo=namedtuple("UndoInfo",["move","captured_piece","was_promoted","old_last_capture_or_pawn"])

# --- Internal Board Representation ---
class InternalBoard:
    # (Class definition remains the same as previous version)
    def __init__(self,board_array=None,turn=1,move_count=0,last_capture_or_pawn=0):
        self.board=[row[:] for row in board_array] if board_array else [[EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.turn=turn; self.move_count=move_count; self.last_capture_or_pawn=last_capture_or_pawn
        self.king_pos={1:None,-1:None}; self._update_king_positions()
    def _update_king_positions(self):
        self.king_pos={1:None,-1:None}
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece=self.board[r][c]
                if piece==wK: self.king_pos[1]=(r,c)
                elif piece==bK: self.king_pos[-1]=(r,c)
    def get_piece_at(self,r,c): return self.board[r][c] if 0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE else None
    def set_piece_at(self,r,c,piece):
        if 0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE:
            current_piece=self.board[r][c]
            if current_piece==wK: self.king_pos[1]=None
            elif current_piece==bK: self.king_pos[-1]=None
            self.board[r][c]=piece
            if piece==wK: self.king_pos[1]=(r,c)
            elif piece==bK: self.king_pos[-1]=(r,c)
    def copy(self):
        new_board=InternalBoard(self.board,self.turn,self.move_count,self.last_capture_or_pawn)
        new_board.king_pos=self.king_pos.copy(); return new_board
    def get_hash(self): return (tuple(tuple(row) for row in self.board),self.turn)
    def make_move(self,move):
        (r1,c1),(r2,c2)=move; piece=self.get_piece_at(r1,c1)
        if piece==EMPTY: return None
        captured_piece=self.get_piece_at(r2,c2); was_promoted=False
        undo_info=UndoInfo(move,captured_piece,False,self.last_capture_or_pawn)
        self.last_capture_or_pawn=0 if abs(piece)==wP or captured_piece!=EMPTY else self.last_capture_or_pawn+1
        self.set_piece_at(r2,c2,piece); self.set_piece_at(r1,c1,EMPTY)
        promoted_to=piece
        if piece==wP and r2==0: promoted_to=wJ; was_promoted=True
        elif piece==bP and r2==5: promoted_to=bJ; was_promoted=True
        if was_promoted: self.set_piece_at(r2,c2,promoted_to); undo_info=undo_info._replace(was_promoted=True)
        self.turn*=-1; self.move_count+=1; return undo_info
    def unmake_move(self,undo_info):
        (r1,c1),(r2,c2)=undo_info.move; moved_piece=self.get_piece_at(r2,c2)
        self.turn*=-1; self.move_count-=1; self.last_capture_or_pawn=undo_info.old_last_capture_or_pawn
        original_piece=wP if self.turn==1 else bP if undo_info.was_promoted else moved_piece
        self.set_piece_at(r1,c1,original_piece); self.set_piece_at(r2,c2,undo_info.captured_piece)

# --- Internal Move Generation ---
# (Functions remain the same: is_valid, generate_pawn..., is_square_attacked..., get_all_valid..., get_capture...)
def is_valid(r,c): return 0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE
def generate_pawn_moves_internal(iboard,r,c,side):
    moves=[]; direction=-1 if side==1 else 1; start_rank=4 if side==1 else 1
    r_f1=r+direction
    if is_valid(r_f1,c) and iboard.get_piece_at(r_f1,c)==EMPTY:
        moves.append(((r,c),(r_f1,c)))
        if r==start_rank: r_f2=r+2*direction;
        if is_valid(r_f2,c) and iboard.get_piece_at(r_f2,c)==EMPTY: moves.append(((r,c),(r_f2,c)))
    for dc in [-1,1]:
        r_cap,c_cap=r+direction,c+dc
        if is_valid(r_cap,c_cap): target=iboard.get_piece_at(r_cap,c_cap);
        if target!=EMPTY and PIECE_COLORS.get(target)==-side: moves.append(((r,c),(r_cap,c_cap))) # Check PIECE_COLORS.get(target)
    return moves
def generate_sliding_moves_internal(iboard,r,c,side,directions):
    moves=[]
    for dr,dc in directions:
        nr,nc=r+dr,c+dc
        while is_valid(nr,nc):
            target=iboard.get_piece_at(nr,nc)
            if target==EMPTY: moves.append(((r,c),(nr,nc)))
            elif PIECE_COLORS.get(target)==-side: moves.append(((r,c),(nr,nc))); break # Check PIECE_COLORS.get(target)
            else: break
            nr+=dr; nc+=dc
    return moves
def generate_knight_moves_internal(iboard,r,c,side):
    moves=[]; offsets=[(1,-2),(2,-1),(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2)]
    for dr,dc in offsets:
        nr,nc=r+dr,c+dc
        if is_valid(nr,nc): target=iboard.get_piece_at(nr,nc);
        if target==EMPTY or PIECE_COLORS.get(target)==-side: moves.append(((r,c),(nr,nc))) # Check PIECE_COLORS.get(target)
    return moves
def generate_king_moves_internal(iboard,r,c,side):
    moves=[]; offsets=[(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
    for dr,dc in offsets:
        nr,nc=r+dr,c+dc
        if is_valid(nr,nc): target=iboard.get_piece_at(nr,nc);
        if target==EMPTY or PIECE_COLORS.get(target)==-side: moves.append(((r,c),(nr,nc))) # Check PIECE_COLORS.get(target)
    return moves
def generate_star_moves_internal(iboard,r,c,side):
    moves=[]; offsets=[(1,1),(-1,1),(1,-1),(-1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    for dr,dc in offsets:
        nr,nc=r+dr,c+dc
        if is_valid(nr,nc): target=iboard.get_piece_at(nr,nc);
        if target==EMPTY or PIECE_COLORS.get(target)==-side: moves.append(((r,c),(nr,nc))) # Check PIECE_COLORS.get(target)
    return moves
def generate_joker_moves_internal(iboard,r,c,side):
    moves=[]; offsets=[(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,2),(2,2),(2,0),(2,-2),(0,-2),(-2,-2),(-2,0),(-2,2)]
    for dr,dc in offsets:
        nr,nc=r+dr,c+dc
        if is_valid(nr,nc): target=iboard.get_piece_at(nr,nc);
        if target==EMPTY or PIECE_COLORS.get(target)==-side: moves.append(((r,c),(nr,nc))) # Check PIECE_COLORS.get(target)
    return moves
def is_square_attacked_internal(iboard,r,c,attacker_side):
    pawn_direction=1 if attacker_side==1 else -1
    for dc in [-1,1]: pr,pc=r+pawn_direction,c+dc;
    if is_valid(pr,pc) and iboard.get_piece_at(pr,pc)==(wP*attacker_side): return True
    for dr,dc in [(1,-2),(2,-1),(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2)]: nr,nc=r+dr,c+dc;
    if is_valid(nr,nc) and iboard.get_piece_at(nr,nc)==(wN*attacker_side): return True
    for dr,dc in [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]: kr,kc=r+dr,c+dc;
    if is_valid(kr,kc) and iboard.get_piece_at(kr,kc)==(wK*attacker_side): return True
    for dr,dc in [(1,1),(-1,1),(1,-1),(-1,-1),(2,0),(-2,0),(0,2),(0,-2)]: sr,sc=r+dr,c+dc;
    if is_valid(sr,sc) and iboard.get_piece_at(sr,sc)==(wS*attacker_side): return True
    for dr,dc in [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,2),(2,2),(2,0),(2,-2),(0,-2),(-2,-2),(-2,0),(-2,2)]: jr,jc=r+dr,c+dc;
    if is_valid(jr,jc) and iboard.get_piece_at(jr,jc)==(wJ*attacker_side): return True
    rook_directions=[(0,1),(0,-1),(1,0),(-1,0)]; bishop_directions=[(1,1),(1,-1),(-1,1),(-1,-1)]
    for dr,dc in rook_directions+bishop_directions:
        nr,nc=r+dr,c+dc
        while is_valid(nr,nc):
            target=iboard.get_piece_at(nr,nc)
            if target!=EMPTY:
                target_color = PIECE_COLORS.get(target) # Use get for safety
                if target_color == attacker_side:
                    target_type=abs(target); is_rook_or_queen=target_type in [wR,wQ]; is_bishop_or_queen=target_type in [wB,wQ]
                    is_rook_direction=dr==0 or dc==0; is_bishop_direction=abs(dr)==abs(dc)
                    if (is_rook_direction and is_rook_or_queen) or (is_bishop_direction and is_bishop_or_queen): return True
                break # Blocked by a piece (either color)
            nr+=dr; nc+=dc
    return False
def get_all_valid_moves_internal(iboard,side):
    pseudo_legal_moves=[]; king_pos=iboard.king_pos[side]
    if king_pos is None: return []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece=iboard.get_piece_at(r,c)
            if piece!=EMPTY and PIECE_COLORS.get(piece)==side: # Use get
                piece_type=abs(piece)
                if piece_type==wP: pseudo_legal_moves.extend(generate_pawn_moves_internal(iboard,r,c,side))
                elif piece_type==wN: pseudo_legal_moves.extend(generate_knight_moves_internal(iboard,r,c,side))
                elif piece_type==wK: pseudo_legal_moves.extend(generate_king_moves_internal(iboard,r,c,side))
                elif piece_type==wS: pseudo_legal_moves.extend(generate_star_moves_internal(iboard,r,c,side))
                elif piece_type==wJ: pseudo_legal_moves.extend(generate_joker_moves_internal(iboard,r,c,side))
                else:
                    directions=[]
                    if piece_type in [wR,wQ]: directions.extend([(0,1),(0,-1),(1,0),(-1,0)])
                    if piece_type in [wB,wQ]: directions.extend([(1,1),(1,-1),(-1,1),(-1,-1)])
                    pseudo_legal_moves.extend(generate_sliding_moves_internal(iboard,r,c,side,directions))
    legal_moves=[]
    for move in pseudo_legal_moves:
        undo_info=iboard.make_move(move)
        # Need to check the king position *after* the move was made
        current_king_pos = iboard.king_pos[side] # Side is the player who just moved
        is_in_check = True # Assume check if king disappears
        if current_king_pos is not None:
            # Check if the king of the player who just moved is attacked by the opponent
            is_in_check = is_square_attacked_internal(iboard, current_king_pos[0], current_king_pos[1], -side)

        iboard.unmake_move(undo_info) # IMPORTANT: Unmake the move
        if not is_in_check: legal_moves.append(move)
    return legal_moves
def get_capture_moves_internal(iboard,side):
    legal_moves=get_all_valid_moves_internal(iboard,side); capture_moves=[]
    for move in legal_moves: r2,c2=move[1];
    # Check the piece at the destination square *before* the move is made
    if iboard.get_piece_at(r2,c2)!=EMPTY: capture_moves.append(move)
    return capture_moves

# --- Bot Class ---
class Bot:
    """ Bot using Internal Board, Make/Unmake, IDDFS, AB, QSearch, TT, Single-Threaded """
    def __init__(self,max_depth=10,time_limit=0.095,scores_dict=SCORES_DICT,piece_tables=PIECE_TABLES):
        self.max_depth=max_depth; self.time_limit=time_limit; self.calculation_time=0
        self.nodes_visited=0; self.q_nodes_visited=0; self.tt_hits=0
        self.SCORES_DICT={k:v for k,v in scores_dict.items() if k!=EMPTY}
        self.PIECE_TABLES={abs(k):v for k,v in piece_tables.items()}
        self.transposition_table={}
        # Removed max_threads

    def _translate_external_to_internal(self,external_board):
        # (Implementation remains the same)
        board_array=[[EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        try:
            ext_state=external_board.get_board_state()
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE): board_array[r][c]=EXTERNAL_TO_INTERNAL[ext_state[r][c]]
        except Exception as e: return None
        turn=1 if external_board.turn=='white' else -1; move_count=external_board.num_moves
        last_capture_or_pawn=external_board.last_captured
        return InternalBoard(board_array,turn,move_count,last_capture_or_pawn)

    def evaluate_board_internal(self,iboard):
        # (Implementation remains the same)
        evaluation=0; white_king_found=False; black_king_found=False
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece=iboard.board[r][c]
                if piece!=EMPTY:
                    if piece==wK: white_king_found=True
                    if piece==bK: black_king_found=True
                    piece_type=abs(piece); piece_color=PIECE_COLORS.get(piece, 0) # Use get
                    piece_value=self.SCORES_DICT.get(piece,0); positional_bonus=0
                    table=self.PIECE_TABLES.get(piece_type)
                    if table: y,x=r,c; mirrored_y=(BOARD_SIZE-1)-y;
                    positional_bonus=table[y][x] if piece_color==1 else table[mirrored_y][x]
                    total_value=piece_value+(positional_bonus*piece_color); evaluation+=total_value
        if not white_king_found: return -30000
        if not black_king_found: return 30000
        if iboard.last_capture_or_pawn>=FIFTY_MOVE_LIMIT: return 0
        return evaluation

    def _get_move_score_mvv_lva_internal(self,iboard,move):
        # (Implementation remains the same)
        (r1,c1),(r2,c2)=move; attacker_piece=iboard.get_piece_at(r1,c1); victim_piece=iboard.get_piece_at(r2,c2)
        if victim_piece!=EMPTY and attacker_piece!=EMPTY: victim_score=abs(self.SCORES_DICT.get(victim_piece,0)); attacker_score=abs(self.SCORES_DICT.get(attacker_piece,0)); return victim_score*10-attacker_score
        return -1

    def _get_ordered_moves_internal(self,iboard,side):
        # (Implementation remains the same)
        moves=get_all_valid_moves_internal(iboard,side); captures=[]; non_captures=[]
        for move in moves: r2,c2=move[1];
        if iboard.get_piece_at(r2,c2)!=EMPTY: score=self._get_move_score_mvv_lva_internal(iboard,move); captures.append((move,score))
        else: non_captures.append(move)
        captures.sort(key=lambda item:item[1],reverse=True); sorted_capture_moves=[move for move,score in captures]
        random.shuffle(non_captures); return sorted_capture_moves+non_captures

    def quiescence_search(self,iboard,depth,alpha,beta):
        # (Implementation remains the same)
        self.q_nodes_visited+=1; current_side=iboard.turn
        stand_pat_score=self.evaluate_board_internal(iboard)
        if current_side==1:
            if stand_pat_score>=beta: return beta
            alpha=max(alpha,stand_pat_score)
        else:
            if stand_pat_score<=alpha: return alpha
            beta=min(beta,stand_pat_score)
        if depth<=0: return stand_pat_score
        capture_moves=get_capture_moves_internal(iboard,current_side)
        ordered_captures = sorted(capture_moves, key=lambda m: self._get_move_score_mvv_lva_internal(iboard, m), reverse=True)
        if not ordered_captures: return stand_pat_score
        best_score=stand_pat_score
        for move in ordered_captures:
            undo_info=iboard.make_move(move)
            score=self.quiescence_search(iboard,depth-1,alpha,beta)
            iboard.unmake_move(undo_info)
            if current_side==1: best_score=max(best_score,score); alpha=max(alpha,score)
            else: best_score=min(best_score,score); beta=min(beta,score)
            if alpha>=beta: break
        return best_score

    def minimax(self,iboard,depth,alpha,beta):
        # (Implementation remains the same)
        self.nodes_visited+=1; original_alpha=alpha; board_hash=iboard.get_hash()
        tt_entry=self.transposition_table.get(board_hash)
        if tt_entry and tt_entry['depth']>=depth:
            self.tt_hits+=1
            if tt_entry['flag']==TT_EXACT: return tt_entry['score']
            elif tt_entry['flag']==TT_LOWERBOUND: alpha=max(alpha,tt_entry['score'])
            elif tt_entry['flag']==TT_UPPERBOUND: beta=min(beta,tt_entry['score'])
            if alpha>=beta: return tt_entry['score']
        white_king_found=iboard.king_pos[1] is not None; black_king_found=iboard.king_pos[-1] is not None
        is_draw=iboard.last_capture_or_pawn>=FIFTY_MOVE_LIMIT; is_terminal=not white_king_found or not black_king_found or is_draw
        if is_terminal: return self.evaluate_board_internal(iboard)
        if depth<=0: return self.quiescence_search(iboard,MAX_QUIESCENCE_DEPTH,alpha,beta)
        current_side=iboard.turn; moves=self._get_ordered_moves_internal(iboard,current_side)
        if not moves: return self.evaluate_board_internal(iboard)
        best_score=float('-inf') if current_side==1 else float('inf')
        for move in moves:
            undo_info=iboard.make_move(move)
            score=self.minimax(iboard,depth-1,alpha,beta)
            iboard.unmake_move(undo_info)
            if current_side==1: best_score=max(best_score,score); alpha=max(alpha,score)
            else: best_score=min(best_score,score); beta=min(beta,score)
            if alpha>=beta: break
        flag=TT_EXACT
        if best_score<=original_alpha: flag=TT_UPPERBOUND
        elif best_score>=beta: flag=TT_LOWERBOUND
        self.transposition_table[board_hash]={'depth':depth,'score':best_score,'flag':flag}
        return best_score

    # Removed _evaluate_move_task_internal function

    def get_best_move_at_depth(self,internal_board,side,depth):
        """ Finds best move for a specific depth (Single-Threaded). """
        best_moves=[]; best_value=float('-inf') if side==1 else float('inf')
        alpha=float('-inf'); beta=float('inf')
        moves=self._get_ordered_moves_internal(internal_board,side)
        if not moves: return None,float('-inf')

        # Handle single move case separately
        if len(moves)==1:
             move=moves[0]
             undo_info=internal_board.make_move(move)
             score=self.minimax(internal_board,depth-1,alpha,beta)
             internal_board.unmake_move(undo_info)
             return move,score

        # --- Sequential Evaluation Loop (Replaces ThreadPoolExecutor) ---
        for move in moves:
            # Make move, search, unmake move
            undo_info = internal_board.make_move(move)
            # Call minimax for the opponent's turn after the move
            move_value = self.minimax(internal_board, depth - 1, alpha, beta)
            internal_board.unmake_move(undo_info) # Backtrack

            # Keep track of the best move for 'side'
            if side == 1: # White (Maximizing)
                if move_value > best_value:
                    best_value = move_value
                    best_moves = [move]
                    alpha = max(alpha, best_value) # Update alpha
                elif move_value == best_value:
                    best_moves.append(move)
            else: # Black (Minimizing)
                if move_value < best_value:
                    best_value = move_value
                    best_moves = [move]
                    beta = min(beta, best_value) # Update beta
                elif move_value == best_value:
                    best_moves.append(move)

            # Alpha-Beta Pruning at the root level
            if alpha >= beta:
                break
        # --- End Sequential Evaluation ---

        selected_move=None
        if best_moves: selected_move=random.choice(best_moves)
        elif moves: selected_move=random.choice(moves); best_value=float('-inf') if side==1 else float('inf')
        return selected_move,best_value

    # --- Main move function --- (Unchanged from previous single-threaded version)
    def move(self,side,external_board):
        self.start_time_for_move=time.time(); self.calculation_time=0; self.nodes_visited=0; self.q_nodes_visited=0; self.tt_hits=0; best_move_overall_internal=None
        try:
            internal_board=self._translate_external_to_internal(external_board);
            if internal_board is None: raise ValueError("Board translation failed")
            internal_side=internal_board.turn
            if (side=='white' and internal_side!=1) or (side=='black' and internal_side!=-1): pass
        except Exception as e: possible_moves=external_board.get_all_valid_moves(side); return random.choice(possible_moves) if possible_moves else None
        initial_moves_internal=get_all_valid_moves_internal(internal_board,internal_side)
        if not initial_moves_internal: return None
        white_king_found=internal_board.king_pos[1] is not None; black_king_found=internal_board.king_pos[-1] is not None
        is_draw=internal_board.last_capture_or_pawn>=FIFTY_MOVE_LIMIT
        if not white_king_found or not black_king_found or is_draw: return None
        self.transposition_table.clear(); last_completed_depth=0; last_score=float('-inf') if internal_side==1 else float('inf')
        for depth in range(1,self.max_depth+1):
            time_elapsed=time.time()-self.start_time_for_move; time_remaining=self.time_limit-time_elapsed
            if time_remaining<(self.time_limit*0.05) or time_elapsed>self.time_limit*0.98: break
            try:
                current_best_move_internal,current_best_score=self.get_best_move_at_depth(internal_board,internal_side,depth)
                time_elapsed_after=time.time()-self.start_time_for_move
                if time_elapsed_after>=self.time_limit*0.99: break
                if current_best_move_internal is not None: best_move_overall_internal=current_best_move_internal; last_completed_depth=depth; last_score=current_best_score
                if abs(current_best_score)>=30000: break
            except Exception as e: break
        self.calculation_time=time.time()-self.start_time_for_move; final_move_external=None
        if best_move_overall_internal: final_move_external=best_move_overall_internal
        else:
            if initial_moves_internal: final_move_external=random.choice(initial_moves_internal)
        print(f"Bot({side}): {final_move_external}. D:{last_completed_depth} S:{last_score:.0f} N:{self.nodes_visited} QN:{self.q_nodes_visited} TT:{self.tt_hits} T:{self.calculation_time:.4f}s")
        return final_move_external

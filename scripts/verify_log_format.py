#!/usr/bin/env python3
"""
Verify that a game log matches the Toss or Hold'em format expected by our implementation.
This helps ensure the C++ solver generates data compatible with actual gameplay.
"""

import re
from typing import List, Tuple, Optional


def parse_card(card_str: str) -> Optional[int]:
    """
    Convert card string (e.g., 'Ks', '8h', '2c') to card index (0-51).
    
    Card encoding: rank (0-12: 2-A) * 4 + suit (0-3: c, d, h, s)
    Ranks: 2=0, 3=1, ..., K=11, A=12
    Suits: c=0, d=1, h=2, s=3
    """
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    
    if len(card_str) < 2:
        return None
    
    rank_char = card_str[0]
    suit_char = card_str[1].lower()
    
    if rank_char not in rank_map or suit_char not in suit_map:
        return None
    
    return rank_map[rank_char] * 4 + suit_map[suit_char]


def verify_log_format(log_text: str) -> dict:
    """
    Verify a game log matches expected format.
    Returns dict with verification results.
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'rounds': 0,
        'game_flow': []
    }
    
    lines = log_text.strip().split('\n')
    current_round = None
    current_street = None
    board_cards = []
    discard_count = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # Round header
        if line.startswith('Round #'):
            match = re.match(r'Round #(\d+), A \((\d+)\), B \((\d+)\)', line)
            if match:
                results['rounds'] += 1
                current_round = int(match.group(1))
                results['game_flow'].append(f"Round {current_round} started")
                board_cards = []
                discard_count = 0
                current_street = 'preflop'
            else:
                results['errors'].append(f"Line {i}: Invalid round header: {line}")
        
        # Blinds
        elif 'posts the blind' in line:
            results['game_flow'].append("Blinds posted")
        
        # Cards dealt
        elif 'dealt' in line:
            match = re.search(r'\[([^\]]+)\]', line)
            if match:
                cards_str = match.group(1)
                cards = [c.strip() for c in cards_str.split()]
                if len(cards) != 3:
                    results['errors'].append(f"Line {i}: Expected 3 pre-discard cards, got {len(cards)}")
                else:
                    # Verify cards are valid
                    for card in cards:
                        if parse_card(card) is None:
                            results['errors'].append(f"Line {i}: Invalid card format: {card}")
                    results['game_flow'].append(f"Cards dealt: {len(cards)} cards")
            else:
                results['errors'].append(f"Line {i}: Could not parse dealt cards")
        
        # Flop
        elif line.startswith('Flop'):
            match = re.search(r'\[([^\]]+)\]', line)
            if match:
                cards_str = match.group(1)
                cards = [c.strip() for c in cards_str.split(',')]
                if len(cards) != 2:
                    results['errors'].append(f"Line {i}: Expected 2 flop cards, got {len(cards)}")
                else:
                    board_cards = cards
                    current_street = 'flop'
                    results['game_flow'].append(f"Flop: {len(cards)} cards")
        
        # Discard
        elif 'discards' in line.lower():
            discard_count += 1
            match = re.search(r'discards (\w+)', line, re.IGNORECASE)
            if match:
                card = match.group(1)
                if parse_card(card) is None:
                    results['errors'].append(f"Line {i}: Invalid discard card: {card}")
                results['game_flow'].append(f"Discard {discard_count}: {card}")
        
        # Discard phase board
        elif line.startswith('Discard'):
            match = re.search(r'\[([^\]]+)\]', line)
            if match:
                cards_str = match.group(1)
                cards = [c.strip() for c in cards_str.split(',')]
                expected = 2 + discard_count  # 2 flop + discards
                if len(cards) != expected:
                    results['warnings'].append(
                        f"Line {i}: Discard phase board has {len(cards)} cards, expected {expected}"
                    )
                board_cards = cards
        
        # Turn
        elif line.startswith('Turn'):
            match = re.search(r'\[([^\]]+)\]', line)
            if match:
                cards_str = match.group(1)
                cards = [c.strip() for c in cards_str.split(',')]
                if len(cards) != 5:
                    results['errors'].append(f"Line {i}: Expected 5 cards on turn, got {len(cards)}")
                else:
                    current_street = 'turn'
                    results['game_flow'].append("Turn")
        
        # River
        elif line.startswith('River'):
            match = re.search(r'\[([^\]]+)\]', line)
            if match:
                cards_str = match.group(1)
                cards = [c.strip() for c in cards_str.split(',')]
                if len(cards) != 6:
                    results['errors'].append(f"Line {i}: Expected 6 cards on river, got {len(cards)}")
                else:
                    current_street = 'river'
                    results['game_flow'].append("River (6 cards total)")
        
        # Showdown
        elif 'shows' in line:
            match = re.search(r'shows \[([^\]]+)\]', line)
            if match:
                cards_str = match.group(1)
                cards = [c.strip() for c in cards_str.split()]
                if len(cards) != 2:
                    results['errors'].append(f"Line {i}: Expected 2 post-discard cards, got {len(cards)}")
                else:
                    results['game_flow'].append("Showdown")
        
        # Awards
        elif 'awarded' in line:
            results['game_flow'].append("Round complete")
    
    # Final verification
    if discard_count != 2:
        results['warnings'].append(f"Expected 2 discards total, found {discard_count}")
    
    if results['errors']:
        results['valid'] = False
    
    return results


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_log_format.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    with open(log_file, 'r') as f:
        log_text = f.read()
    
    results = verify_log_format(log_text)
    
    print(f"\n{'='*60}")
    print("Log Format Verification")
    print(f"{'='*60}\n")
    
    print(f"Rounds found: {results['rounds']}")
    print(f"\nGame Flow:")
    for step in results['game_flow']:
        print(f"  - {step}")
    
    if results['errors']:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['valid'] and not results['warnings']:
        print(f"\n✅ Log format is valid and matches Toss or Hold'em rules!")
    elif results['valid']:
        print(f"\n⚠️  Log format is mostly valid but has warnings")
    else:
        print(f"\n❌ Log format has errors")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

#bridge generator

import random

import torch
import numpy as np
import itertools
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

"""
## Example: ♠AKQ ♥J53 ♦9 ♣KQJ985

### Current Binary Representation (4x13)
```
         2  3  4  5  6  7  8  9  T  J  Q  K  A
Spades:  0  0  0  0  0  0  0  0  0  0  1  1  1  ← cards present
Hearts:  0  1  0  1  0  0  0  0  0  1  0  0  0
Diamonds:0  0  0  0  0  0  0  1  0  0  0  0  0
Clubs:   0  0  0  1  0  0  1  1  0  1  1  1  0
```
"""

def generate_random_hand():
    tensor = torch.zeros((4, 13), dtype=torch.int)
    indices = random.sample([(c,s) for c in range(4) for s in range(13)], 13)
    for c,s in indices:
        tensor[c,s] = 1.0
    return tensor

def generate_random_hand2():
    deck = torch.zeros((4,13), dtype=torch.float)
    #rnd_indx = torch.randperm(7)
    #rnd_indx = torch.randperm(torch.randint(1,1000,(1,)))
    #torch.manual_seed(torch.randint(1,1000,(1,)))
    rnd_indx = torch.randperm(52)
    chosen_base = torch.randint(3,33,(1,))
    chosen_cards = rnd_indx[chosen_base:chosen_base+13]
    for i,card in enumerate(chosen_cards):
        deck[card//13, card%13] = 1.0
    return deck    


def print_hand(hnds: torch.Tensor, print_hcp = False):
    hnds = hnds.unsqueeze(0) if len(hnds.shape)==2 else hnds
    deck = np.array(["2","3","4","5","6","7","8","9","T","J","Q","K","A"]) 
    for i,h in enumerate(hnds):
        print(f"Hand:{i+1}")
        if print_hcp:
            print(f"HCP: {hcp(h)}")
        for suit in range(4):
            if len(h[suit].nonzero())==0:
                print("----")
            else: #for i,j in enumerate(np.nonzero(h)...if i==suit
                print(''.join([deck[j.item()] for j in np.nonzero(h[suit])])[::-1] )
        print("===\n")

# -----------------------------------------------
def generate_training_set(num_hands):
    hands = [generate_random_hand2() for _ in range(num_hands)] 
    return torch.stack(hands)


def verify_generated_hands(hands: torch.Tensor, expected_shape=(4, 13)) -> None:
    if hands.ndim != 3:
        raise ValueError(f"Expected 3D tensor [N,4,13], got shape {tuple(hands.shape)}")
    if tuple(hands.shape[1:]) != expected_shape:
        raise ValueError(
            f"Expected per-hand shape {expected_shape}, got {tuple(hands.shape[1:])}"
        )
    if not torch.all((hands == 0) | (hands == 1)):
        raise ValueError("Hands must be binary encoded with values in {0, 1}")
    cards_per_hand = hands.sum(dim=(1, 2))
    if not torch.all(cards_per_hand == 13):
        raise ValueError("Each hand must contain exactly 13 cards")

#-----------------------------------------------
def closest_hand(candidate_hand: torch.Tensor, partner_hand: torch.Tensor, OVERLAP_INDX=3) -> torch.Tensor:
    """Change partner_hand, so no overlapping cards will exist between partner_hand and candidate_hand."""

    partner_hand[partner_hand == 1] = 2
    merged_hand = candidate_hand + partner_hand
    
    # indices of overlapping cards
    overlapping_indices = (merged_hand == OVERLAP_INDX).nonzero()

    #for each overlapping index, find the available cards (same suit equal to 0) and replace 
    for idx in overlapping_indices:
        suit, card = idx
        available_cards = (merged_hand[suit] == 0).nonzero()
        if available_cards.numel() > 0:
            replacement_suit = suit 
            #get argmin from end of available_cards
            closest_idx_from_end = (available_cards - card).abs().flip(dims=[0]).argmin()
            closest_idx_from_start = available_cards.size(0) - closest_idx_from_end - 1
            replacement_card = available_cards[closest_idx_from_start]
        else:
            # get a random available card (from another suit, with close rank)
            available_cards = (merged_hand == 0).nonzero()
            replacement_idx = available_cards[(available_cards[:, 1] - card).abs().argmin()]
            replacement_suit = replacement_idx[0]
            replacement_card = replacement_idx[1]
    
        merged_hand[suit, card] = 2 #duplicated card assigned just to partner_hand
        merged_hand[replacement_suit, replacement_card] = 1 #NEW card assigned to candidate_hand

    #merged_hand[merged_hand == 4] = 0 #reset all used non-available cards
    return merged_hand

#-----------------------------------------------
def hcp(hnd):
    return np.sum([c[1].item() for c in hnd[:,8:].nonzero()])


#-----------------------------------------------
def calc_losers(hand: torch.Tensor) -> tuple[float, torch.Tensor]:
    """
        Return total and per-suit losers for a binary [4, 13] hand tensor.
        Ranks run from 2 at index 0 to ace at index 12. Suits with at least
        three cards lose one trick for each missing A/K/Q. Short suits compare
        their held cards with A or A/K.
    """
    suit_losers = torch.zeros(4, dtype=torch.float32, device=hand.device)
    ideal_short_suit_ranks = (3, 2, 1, 0)

    for suit_index, suit in enumerate(hand):
        card_count = int(suit.sum().item())
        if card_count >= 3:
            honors_count = int(suit[-3:].sum().item())
            if honors_count == 1: #single honor card
                honor_weights = [0.5, 0.75, 1.0]
                suit_losers[suit_index] = 3 - torch.dot(suit[-3:].detach().clone().to(torch.float64), torch.tensor(honor_weights, dtype=torch.float64)).sum()
                if (suit[10] == 1 and suit[9] == 1) | (suit[9] == 1 and suit[8] == 1): #QJx or AJT, KJT
                    suit_losers[suit_index] -= 0.5
            elif honors_count == 2: #two honor cards
                suit_losers[suit_index] = 3 - suit[-3:].detach().clone().to(torch.float64).sum()
                if (suit[10]==1 and suit[12]==1): # AQx (and not AQJ that keeps 1 loser)
                    if suit[9]==0: #has no J
                        suit_losers[suit_index] += 0.5
            else: # Do wrong to none
                suit_losers[suit_index] = 3 - suit[-3:].detach().clone().to(torch.float64).sum()
            continue

        if card_count == 2:
            #compare held_ranks gaps with the ideal [12,11]
            checked_card = 0
            honors_held = torch.nonzero(suit[-4:], as_tuple=False).flatten().flip(0)
            # for each ideal_short_suit_ranks element, if present in held_cards, continue. else, add loser count by 1
            for ideal_rank in ideal_short_suit_ranks[:card_count]:
                if ideal_rank not in honors_held and checked_card < card_count+1:
                    if ideal_rank in [3,2]:
                        loser_factor = 1
                    else:
                        loser_factor = 0.5
                    suit_losers[suit_index] += loser_factor
                else:
                    checked_card += 1
                    if suit_losers[suit_index]==1.5:
                        suit_losers[suit_index] = 1.75
        if card_count == 1:
            honors_held = torch.nonzero(suit[-4:], as_tuple=False).flatten().flip(0)
            if 3 not in honors_held:
                suit_losers[suit_index] += 1
            elif 2 in honors_held:
                suit_losers[suit_index] += 0.5
    return float(suit_losers.sum().detach()), suit_losers


def balanced_factor(shp : int ,base)-> torch.Tensor:
    shapev = np.sum([((shp//10**i)%10-base)**4  for i in range(4)])
    return torch.tensor(shapev)

#2. 
def no6M(shp : int):
    vec = 0 if shp%10>5 or shp%100//10>5 else 1.0
    return vec 

def nt_range(hnd, min=15, max=17):
    return 1.0 if (hcp(hnd)>=min and hcp(hnd)<=max) else 0


# ------------------------------------------------------

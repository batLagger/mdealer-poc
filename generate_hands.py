
from hand_emb import generate_smart_hands
from hand_pair_emb import create_query_pair, decode_pair, print_pair2

def generate_hands(hand1_str, hand2_str, hand_set, model, ltc_param=1.0, shape_param=0.1, sorted_shape_param=0.6):
    #ask user for hand1 and hand2 strings 
    #hand1_str = "AT2.AJ8.K9532.KJ"
    #hand2_str = "KQJ9863.74.A.AT6"
    
    if not hand1_str:
        hand1_str = input("Enter hand1 string: ")
    if not hand2_str:
        hand2_str = input("Enter hand2 string: ")
    
    query_pair = create_query_pair(hand1_str, hand2_str)
    south, north = decode_pair(query_pair)

    gen_hand_set =generate_smart_hands(north, south, hand_set, model,ltc_param, shape_param, sorted_shape_param, num_pairs=15)
    print_pair2(gen_hand_set)
    
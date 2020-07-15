
def compare(a1, a2):
    if a1 == a2:
        return 0.5
    else:
        if [a1, a2] in [[0, 1], [1, 2], [2, 0]]:
            return 1
        elif [a1, a2] in [[1, 0], [2, 1], [0, 2]]:
            return 0
        else:
            raise ValueError("Each action must be in {0,1,2}")


def map_num_to_hand(cards):
    dic = {0:"ぐー", 1:"ちょき", 2:"ぱー"}
    return [dic[card] for card in cards]

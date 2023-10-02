import random

cards = ["jack", "queen", "king"]
random.shuffle(cards)
for index, card in enumerate(cards, start=1):
    print(f"{index}. {card}")
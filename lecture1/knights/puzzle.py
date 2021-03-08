from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # Game rules
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    
    # Information that 'A' said can be encoded this way:
    #                                                    And(AKnight, AKnave)
    # If it's true, he is a knight, otherwise he is a knave
    Implication(And(AKnight, AKnave), AKnight),
    Implication(Not(And(AKnight, AKnave)), AKnave)
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # Game rules
    Or(AKnight, AKnave),
    Or(BKnight, BKnave),
    Not(And(AKnight, AKnave)),
    Not(And(BKnight, BKnave)),
    
    # Information that 'A' said can be encoded this way:
    #                                                    And(AKnave, BKnave)
    # If it's true, he is a knight, otherwise he is a knave
    Implication(And(AKnave, BKnave), AKnight),
    Implication(Not(And(AKnave, BKnave)), AKnave)
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # Game rules
    Or(AKnight, AKnave),
    Or(BKnight, BKnave),
    Not(And(AKnight, AKnave)),
    Not(And(BKnight, BKnave)),
    
    # Information that 'A' said can be encoded this way:
    #                                                    Or(And(AKnave, BKnave), And(AKnight, BKnight))
    # If it's true, he is a knight, otherwise he is a knave
    Implication(Or(And(AKnave, BKnave), And(AKnight, BKnight)), AKnight),
    Implication(Not(Or(And(AKnave, BKnave), And(AKnight, BKnight))), AKnave),
    
    # Information that 'B' said can be encoded this way:
    #                                                    Not(Or(And(AKnave, BKnave), And(AKnight, BKnight)))
    # If it's true, he is a knight, otherwise he is a knave
    Implication(Not(Or(And(AKnave, BKnave), And(AKnight, BKnight))), BKnight),
    Implication(Not(Not(Or(And(AKnave, BKnave), And(AKnight, BKnight)))), BKnave)
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # Game rules
    Or(AKnight, AKnave),
    Or(BKnight, BKnave),
    Or(CKnight, CKnave),
    Not(And(AKnight, AKnave)),
    Not(And(BKnight, BKnave)),
    Not(And(CKnight, CKnave)),
    
    # Information that 'A' said can be encoded this way:
    #                                                    *if he said "I'm a knight"*
    #                                                    AKnight
    #                                                    *if he said "I'm a knave"*
    #                                                    AKnave
    # So we gonna join this cases with 'or', and check if he is a knight or knave for both of them
    Or(
        # If he said he's a knight
        And(Implication(AKnight, AKnight), Implication(Not(AKnight), AKnave)),
        # If he said he's a knave
        And(Implication(AKnave, AKnight), Implication(Not(AKnave), AKnave))
    ),
    
    # Information that 'B' said can be encoded this way:
    #                                                    And(Implication(AKnave, AKnight), Implication(Not(AKnave), AKnave)
    #                                                    CKnave
    # If it's true, he is a knight, otherwise he is a knave
    Implication(And(Implication(AKnave, AKnight), Implication(Not(AKnave), AKnave)), BKnight),
    Implication(Not(And(Implication(AKnave, AKnight), Implication(Not(AKnave), AKnave))), BKnave),
    Implication(CKnave, BKnight),
    Implication(Not(CKnave), BKnave),
    
    # Information that 'C' said can be encoded this way:
    #                                                    AKnight
    # If it's true, he is a knight, otherwise he is a knave
    Implication(AKnight, CKnight),
    Implication(Not(AKnight), CKnave)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()

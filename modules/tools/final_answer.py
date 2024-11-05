
# final answer tool
def final_answer(answer: str):
# def final_answer(answer: str, phone_number: str = "", address: str = ""):
    """Returns a natural language response to the user. There are four sections 
    to be returned to the user, those are:
    - `answer`: the final natural language answer to the user's question, should provide as much context as possible.
    - `phone_number`: the phone number of top recommended restaurant (if found).
    - `address`: the address of the top recommended restaurant (if found).
    """
    return {
        "answer": answer
    }
    # return {
    #     "answer": answer,
    #     "phone_number": phone_number,
    #     "address": address,
    # }

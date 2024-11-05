from pydantic import BaseModel




# class to model the reviews
class Rec(BaseModel):
    title: str
    description: str
    comments: list[str]

    def __str__(self):
        """LLM-friendly string representation of the recommendation(s)."""
        return f"Title: {self.title}\nDescription: {self.description}\nComments:\n" + "\n".join(self.comments)
    



# dummy search tool
def search(query: str) -> list[Rec]:
    """Provides access to search reddit. You can use this tool to find restaurants.
    Best results can be found by providing as much context as possible, including
    location, cuisine, and the fact that you're looking for a restaurant, cafe,
    etc.
    """


    dummy_recs = [
        Rec(
            title="Best Pizza in Rome",
            description="Looking for the best pizza in Rome?",
            comments=[
                "JohnDoe (25 upvotes): Try Pizza Roma!",
                "Foodie88 (20 upvotes): Pizza Vesuvio is amazing!",
                "Traveler123 (18 upvotes): Don't miss Pizzeria La Bella! CALL THEM AT +39 06 123456789",
            ],
        ),
        Rec(
            title="Top Restaurants in Tokyo",
            description="Seeking top-rated restaurants in Tokyo?",
            comments=[
                "TokyoFan (30 upvotes): Sushi Saito is incredible!",
                "FoodLover22 (22 upvotes): Try Ramen Ichiran!",
                "AdventureSeeker (20 upvotes): Don't miss Tofuya Ukai!",
            ],
        ),
        Rec(
            title="Best Burgers in NYC",
            description="Looking for the best burgers in NYC?",
            comments=[
                "NYCFoodie (28 upvotes): Shake Shack is a must!",
                "BurgerLover (25 upvotes): Try Five Guys!",
                "Foodie90 (22 upvotes): Don't miss Burger Joint!",
            ],
        ),
    ]


        
    return dummy_recs


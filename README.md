# BookReviews
Utilizing Python and Machine Learning to analyze data and function as a Book Recommender. 

Outline: 
        server folder: All backend logic goes here, everything I need is grouped underneath this text.

        models folder: Contains Book class where everything I fetch, parse, and recommend maps into a Book

        fetcher folder: Contains fetcher class. Keeps HTTP‑requests, rate‑limiting, error‑handling, and raw JSON/HTML separate from parsing or business logic.

        preprocessing folder: Contains preprocessing class. It's responsible for tokenization, lowercasing, stop‑word removal, stemming/lemmatization. I want all descriptions and user inputs in a uniform form before I turn them into vectors. 

        features folder: Contains features class. Turns cleaned text (and other metadata) into numerical or symbolic features. Separates “how do I get a vector out of a Book?” from “how do I compare two vectors?” Allows me to swap in new algorithms  without touching your recommendation logic.

        recommender folder: Contains recommender class and recommendation_engine class. 
        recommendation_engine class is the brains of my system, it computes similarities, ranks, and serve recommendations.
        recommender class is responsible for orchestrating the flow of the project. Take user adjectives → preprocess → extract query vector Ask Fetcher for candidate books → parse → build Book objects Feature‑extract each Book → get book vectors Call RecommendationEngine.rank(...) → return top results etc.


        
4/15/25
 - Added all necessary classes to begin this project. Gave an outline and description for reasoning behind each folder and class. 
 
 
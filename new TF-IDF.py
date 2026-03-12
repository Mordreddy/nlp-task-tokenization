from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    """"Oppenheimer," directed by Christopher Nolan, is a biographical thriller about J. Robert Oppenheimer, the physicist who led the Manhattan Project and developed the atomic bomb. The film stars Cillian Murphy as Oppenheimer and features an ensemble cast including Emily Blunt, Matt Damon, and Robert Downey Jr. It explores the moral dilemmas and political fallout of creating such a destructive weapon. The movie was praised for its intense performances, haunting score, and stunning IMAX cinematography.

"Dune: Part Two" continues the adaptation of Frank Herbert's sci‑fi epic, directed by Denis Villeneuve. The film follows Paul Atreides (Timothée Chalamet) as he unites with the Fremen people to wage war against the Harkonnen oppressors. The cast includes Zendaya, Austin Butler, Florence Pugh, and Christopher Walken. Early reviews highlight its breathtaking visuals, complex political intrigue, and action sequences. It is scheduled for release in March 2024 and is expected to be a major blockbuster.

"The Shawshank Redemption," directed by Frank Darabont, is a drama based on a Stephen King novella. It tells the story of Andy Dufresne, a banker wrongly imprisoned for murder, and his friendship with fellow inmate Ellis "Red" Redding. Over two decades, Andy maintains hope and dignity while planning his escape. The film is renowned for its powerful storytelling, memorable quotes, and uplifting message. It consistently ranks among the greatest films of all time.""",

    """Currently, due to physical discs being shipped out early, a large number of gameplay videos, key plot points, and even the final ending of Resident Evil 9: Requiem have spread like wildfire across various social media platforms. Outlets such as IGN have confirmed that this leak is arguably the most severe in the series' history.
Faced with this spoiler disaster, Capcom issued an urgent statement yesterday with a heavy tone:
"To all players looking forward to Resident Evil: Requiem,
We have confirmed that a large number of gameplay videos of Resident Evil: Requiem have appeared online, uploaded by individuals who are suspected to have obtained the game content through improper means ahead of its release. In order to protect the experience of players who have been eagerly awaiting this title, we kindly ask that you refrain from sharing or making public such videos on social media before the official launch."

With the February 27 release date approaching, players preparing to return to Raccoon City can arrange to pre-load the game in advance. According to information obtained by the Twitter account PlayStationSize from PlayStation server databases, pre-loading for the PS5 version of Resident Evil 9: Requiem will begin on February 25. The game's file size has also been revealed. The PS5 version will require approximately 72.88 GB of storage, which is significantly larger than the same platform's Resident Evil 4 Remake (58.5 GB) and Resident Evil 8: Village (32.6 GB). For players with limited internal storage, it may be necessary to clear up some space in advance or consider expansion options.

The Resident Evil series has evolved from fixed‑camera angles to first‑person horror and action‑oriented remakes. Resident Evil 4 Remake (2023) was critically acclaimed for modernizing the classic while keeping its spirit. Fans now eagerly await news of Resident Evil 9, hoping for a balance between action and horror, and perhaps a conclusion to the Winters family saga. Capcom has not made any official announcement, but job listings suggest the project is well underway.""",

    """On February 21, the Moscow Zoo launched a series of events celebrating Chinese New Year. On the same day, the Chinese Embassy in Russia and the zoo presented New Year gifts to the giant pandas living in Russia.

Zhang Wei, Minister of the Chinese Embassy in Russia, stated at the event that the Spring Festival is the most significant traditional holiday for the Chinese people. Celebrations for the festival are held in many Russian cities, with Moscow's events being particularly grand. The giant pandas in Russia are a vivid symbol of Sino-Russian friendship. Pandas "Ruyi" and "Dingding" have lived in Russia for nearly seven years, and the birth of the panda cub "Katyusha" has become a beautiful symbol of the enduring friendship between China and Russia.

Recently, cultural events themed around the Spring Festival have been held in many places overseas, with the festive atmosphere extending beyond Chinese communities and attracting more and more international audiences to experience Chinese culture and share in the holiday joy. The Spring Festival, recognized as an intangible cultural heritage, is gradually becoming a truly global celebration.

In Australia, the Spring Festival Carnival Arts Festival is currently underway. Ethnic performances such as Dunhuang dance, Sichuan opera face-changing, and dragon and lion dances have taken to the streets of Sydney and theater stages, drawing large crowds.

Chinese New Year, also known as the Spring Festival, is the most important traditional festival in China. It marks the beginning of the lunar new year and usually falls between late January and mid‑February. Celebrations include family reunions, feasts, red envelopes (hongbao) filled with money, fireworks, and dragon dances. Each year is associated with one of the 12 zodiac animals. The festival lasts for 15 days, ending with the Lantern Festival."""
]

doc_names = ["movie", "Resident Evil", "new year"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r'(?u)\b\w+\b',
    use_idf=True,
    smooth_idf=False,
    norm=None
)

tfidf_matrix = vectorizer.fit_transform(documents)

print(f"Sparse matrix shape: {tfidf_matrix.shape}")
print(f"Number of non-zero elements: {tfidf_matrix.nnz}")

feature_names = vectorizer.get_feature_names_out()

# Print the Top-K keywords for each document
TOP_K = 10
for doc_index in range(tfidf_matrix.shape[0]):
    row = tfidf_matrix[doc_index]
    coo = row.tocoo()
    word_weight_pairs = [(feature_names[j], v) for i, j, v in zip(coo.row, coo.col, coo.data)]
    sorted_pairs = sorted(word_weight_pairs, key=lambda x: x[1], reverse=True)

    print(f"\ndoc '{doc_names[doc_index]}' s Top-{TOP_K} keywords：")
    for word, weight in sorted_pairs[:TOP_K]:
        print(f"  word '{word}' : {weight:.4f}")
import requests
import pandas as pd

API_KEY = "a0830d6065f541829878046079abff27"  # Put your key here
companies = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]
all_news = []

for symbol in companies:
    # You can also use company name if you prefer (more results)
    query = symbol
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=20&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    for article in data.get("articles", []):
        all_news.append({
            "symbol": symbol,
            "title": article.get("title"),
            "publisher": article.get("source", {}).get("name"),
            "link": article.get("url"),
            "date": article.get("publishedAt"),
        })

news_df = pd.DataFrame(all_news)
print(news_df.head())
news_df.to_csv("company_news_headlines.csv", index=False)

E_keywords = ["environment", "climate", "pollution", "carbon", "renewable", "sustainability", "emissions", "waste", "eco-friendly", "green"]
S_keywords = ["diversity", "community", "labor", "inclusion", "rights", "equality", "philanthropy", "employee", "workplace", "welfare"]
G_keywords = ["board", "governance", "audit", "corruption", "compliance", "transparency", "regulation", "ethics", "management"]

def keyword_count(text, keywords):
    if pd.isnull(text):
        return 0
    text = text.lower()
    return sum(kw in text for kw in keywords)

news_df['E_score'] = news_df['title'].apply(lambda x: keyword_count(x, E_keywords))
news_df['S_score'] = news_df['title'].apply(lambda x: keyword_count(x, S_keywords))
news_df['G_score'] = news_df['title'].apply(lambda x: keyword_count(x, G_keywords))

company_esg = news_df.groupby('symbol')[['E_score', 'S_score', 'G_score']].sum().reset_index()
print(company_esg)
import matplotlib.pyplot as plt

company_esg.set_index('symbol')[['E_score', 'S_score', 'G_score']].plot(kind='bar', figsize=(10,6))
plt.title("Company ESG Headline Keyword Scores")
plt.ylabel("Count")
plt.show()


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


def get_sentiment(text):
    if pd.isnull(text):
        return 0
    return sia.polarity_scores(text)['compound']

news_df['sentiment'] = news_df['title'].apply(get_sentiment)

company_sentiment = news_df.groupby('symbol')['sentiment'].mean().reset_index()
print(company_sentiment)

for pillar, col in zip(['E', 'S', 'G'], ['E_score', 'S_score', 'G_score']):
    news_df[f'{pillar}_mask'] = news_df[col] > 0

# Environmental sentiment per company:
E_sentiment = news_df[news_df['E_mask']].groupby('symbol')['sentiment'].mean().reset_index().rename(columns={'sentiment': 'E_sentiment'})
S_sentiment = news_df[news_df['S_mask']].groupby('symbol')['sentiment'].mean().reset_index().rename(columns={'sentiment': 'S_sentiment'})
G_sentiment = news_df[news_df['G_mask']].groupby('symbol')['sentiment'].mean().reset_index().rename(columns={'sentiment': 'G_sentiment'})

# Merge into company_esg:
company_esg = company_esg.merge(E_sentiment, on='symbol', how='left')\
                         .merge(S_sentiment, on='symbol', how='left')\
                         .merge(G_sentiment, on='symbol', how='left')
print(company_esg)

import matplotlib.pyplot as plt

company_esg.set_index('symbol')[['E_sentiment', 'S_sentiment', 'G_sentiment']].plot(kind='bar', figsize=(10,6))
plt.title("Company ESG Sentiment Scores")
plt.ylabel("Average Sentiment")
plt.show()

# Example: flag if in the top half for ESG keyword count
for pillar in ['E', 'S', 'G']:
    count_col = f'{pillar}_score'
    sent_col = f'{pillar}_sentiment'
    # High attention: above median
    median_count = company_esg[count_col].median()
    company_esg[f'{pillar}_high_attention'] = company_esg[count_col] > median_count
    # Negative sentiment
    company_esg[f'{pillar}_neg_sent'] = company_esg[sent_col] < 0

for pillar in ['E', 'S', 'G']:
    company_esg[f'{pillar}_risk_flag'] = company_esg[f'{pillar}_high_attention'] & company_esg[f'{pillar}_neg_sent']

# Optional: Combine into an "overall risk" flag (any pillar flagged)
company_esg['any_esg_risk'] = company_esg[[f'{pillar}_risk_flag' for pillar in ['E','S','G']]].any(axis=1)

flagged = company_esg[company_esg['any_esg_risk']]
print("Companies flagged as potential high ESG risk:")
print(flagged[['symbol', 'E_score', 'E_sentiment', 'E_risk_flag',
               'S_score', 'S_sentiment', 'S_risk_flag',
               'G_score', 'G_sentiment', 'G_risk_flag']])

import matplotlib.pyplot as plt

# Count how many pillars are flagged for each company
company_esg['num_risk_flags'] = (
    company_esg[['E_risk_flag', 'S_risk_flag', 'G_risk_flag']].sum(axis=1)
)

# Only show companies with at least one risk flag
flagged = company_esg[company_esg['num_risk_flags'] > 0]

# Sort by number of risk flags (optional)
flagged = flagged.sort_values('num_risk_flags', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(flagged['symbol'], flagged['num_risk_flags'], color='crimson')
plt.xlabel('Company')
plt.ylabel('Number of ESG Risk Flags (0–3)')
plt.title('Companies Flagged as High ESG Risk')
plt.ylim(0, 3)
plt.tight_layout()
plt.show()

flagged_pillars = flagged.set_index('symbol')[['E_risk_flag', 'S_risk_flag', 'G_risk_flag']].astype(int)
flagged_pillars.plot(
    kind='bar', 
    stacked=True, 
    figsize=(10,6), 
    color=['green', 'orange', 'blue']
)
plt.xlabel('Company')
plt.ylabel('Risk Flagged (1 = High ESG Risk)')
plt.title('Breakdown of ESG Risk Flags by Pillar')
plt.legend(['Environmental', 'Social', 'Governance'])
plt.tight_layout()
plt.show()


import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Provided transcripts
transcripts = [
     "\"I wanted to express my gratitude for the amazing service I received!\"",
    "\"Your team's responsiveness and professionalism exceeded my expectations.\"",
    "\"I'm thoroughly impressed with the quality of your product!\"",
    "\"Thank you for making my shopping experience so delightful!\"",
    "\"I'm thrilled with the level of care and attention I received from your support team.\"",
    "\"Your company truly understands the meaning of excellent customer service.\"",
    "\"I just wanted to let you know how happy I am with my recent purchase!\"",
    "\"Kudos to your team for going above and beyond to meet my needs!\"",
    "\"The efficiency and speed of your delivery service are outstanding!\"",
    "\"I'm so glad I chose your company for my needs. Keep up the fantastic work!\"",
    "\"Your product has made my life so much easier. Thank you!\"",
    "\"I'm amazed by the attention to detail and craftsmanship of your product.\"",
    "\"Thank you for consistently delivering top-notch service!\"",
    "\"Your company sets the standard for excellence in customer care.\"",
    "\"I'm a loyal customer because of the exceptional service you provide.\"",
    "\"Your team's dedication to customer satisfaction is truly remarkable.\"",
    "\"I can't thank you enough for your prompt and efficient assistance!\"",
    "\"Your product has exceeded all my expectations. I'm a happy customer!\"",
    "\"I've never experienced such exceptional service. You've gained a lifelong customer!\"",
    "\"The ease of use and functionality of your product are simply outstanding.\"",
    "\"I'm so impressed with the level of professionalism displayed by your team.\"",
    "\"Your company has earned my trust and loyalty with your exceptional service.\"",
    "\"Thank you for making me feel valued as a customer. You guys are amazing!\"",
    "\"I'm thoroughly satisfied with my experience with your company. Well done!\"",
    "\"Your team's friendliness and helpfulness have made all the difference!\"",
    "\"I'm grateful for the care and attention your team puts into every interaction.\"",
    "\"Your product has truly improved the quality of my life. Thank you!\"",
    "\"I'm blown away by the efficiency and effectiveness of your customer service.\"",
    "\"Your company's commitment to excellence is evident in every interaction.\"",
    "\"I'm beyond impressed with the level of service I received. You guys are the best!\"",
    "\"Your team's dedication to going the extra mile has not gone unnoticed. Thank you!\"",
    "\"I'm so pleased with my purchase. Your product exceeded my expectations!\"",
    "\"Your company sets the gold standard for customer satisfaction. Well done!\"",
    "\"Thank you for making every interaction with your company a positive one!\"",
    "\"I'm so grateful for the attention to detail and care your team provides.\"",
    "\"Your product has made such a difference in my life. I can't thank you enough!\"",
    "\"I'm thoroughly satisfied with the quality and reliability of your product.\"",
    "\"Your team's professionalism and expertise are truly impressive!\"",
    "\"I'm so glad I chose your company. Your exceptional service speaks for itself!\"",
    "\"Your company has won me over with your outstanding service. Thank you!\"",
    "\"Your product has exceeded my expectations in every way possible.\"",
    "\"I'm so impressed with the level of care and attention your team provides.\"",
    "\"Thank you for making my experience with your company a memorable one!\"",
    "\"Your team's dedication to customer satisfaction is second to none.\"",
    "\"I'm thoroughly satisfied with the level of service I received. Keep up the great work!\"",
    "\"Your product has become an essential part of my daily life. Thank you!\"",
    "\"I'm so grateful for the exceptional service your team provides!\"",
    "\"Your company's commitment to excellence is truly commendable!\"",
    "\"Thank you for making every interaction with your company a positive one!\"",
    "\"Your team's dedication to going above and beyond is truly remarkable. Thank you!\"",

    "\"I'm disappointed with the poor quality of your product.\"",
    "\"Your team's lack of responsiveness has left me frustrated.\"",
    "\"I'm dissatisfied with the level of service I received from your company.\"",
    "\"The constant issues with your product have made me reconsider my loyalty.\"",
    "\"Your company's customer service fell short of my expectations.\"",
    "\"I'm frustrated with the lack of communication regarding my order.\"",
    "\"The product I received was not as described. I'm highly disappointed.\"",
    "\"Your team's lack of accountability has soured my experience.\"",
    "\"I regret choosing your company for my needs. The service was subpar.\"",
    "\"The product I purchased did not meet the promised standards. Very disappointed.\"",
    "\"Your company's inability to resolve my issue in a timely manner is unacceptable.\"",
    "\"I'm dissatisfied with the poor handling of my complaint.\"",
    "\"The constant errors in billing have made me lose trust in your company.\"",
    "\"Your team's lack of empathy and understanding left me feeling undervalued.\"",
    "\"I'm disappointed with the lack of professionalism displayed by your team.\"",
    "\"The product I received was damaged, and the return process was a nightmare.\"",
    "\"Your company's lack of follow-through on promises has left me disillusioned.\"",
    "\"I'm frustrated with the recurring technical issues with your product.\"",
    "\"The customer service I received was rude and unhelpful.\"",
    "\"Your company's failure to meet delivery deadlines has inconvenienced me greatly.\"",
    "\"I'm disappointed with the lack of product support provided.\"",
    "\"The product I received was not worth the price I paid. Very dissatisfied.\"",
    "\"Your team's lack of knowledge about the product was concerning.\"",
    "\"I'm frustrated with the difficulty I faced in reaching your customer service.\"",
    "\"Your company's lack of transparency regarding pricing is deceptive.\"",
    "\"I regret purchasing your product. It did not live up to expectations.\"",
    "\"The constant issues with your website have made it impossible to shop.\"",

    "\"My experience with your company was neither exceptional nor disappointing.\"",
    "\"The service provided was adequate, but nothing particularly stood out.\"",
    "\"I have mixed feelings about my recent interaction with your customer service.\"",
    "\"Overall, my experience with your product was average.\"",
    "\"There were both positive and negative aspects to my experience with your company.\"",
    "\"I neither loved nor hated my experience with your company.\"",
    "\"My feelings about my recent purchase are fairly neutral.\"",
    "\"I have no strong opinions about the service provided.\"",
    "\"My interaction with your company left me feeling indifferent.\"",
    "\"I'm neither overly satisfied nor dissatisfied with my experience.\"",
    "\"The product I received met my basic expectations but didn't exceed them.\"",
    "\"I don't have much to say about my recent experience with your company.\"",
    "\"My feelings about my recent purchase are neither positive nor negative.\"",
    "\"The service provided was neither exceptional nor subpar.\"",
    "\"I have no strong feelings one way or the other about my recent interaction with your company.\"",
    "\"My experience with your company was fairly unremarkable.\"",
    "\"Overall, my experience with your product was neither impressive nor disappointing.\"",
    "\"The service provided was satisfactory, but there's room for improvement.\"",
    "\"I don't have any strong feelings about my recent purchase.\"",
    "\"My interaction with your company was neither outstanding nor terrible.\"",
    "\"My experience with your product was average, with no major complaints.\"",
    "\"There were aspects of my experience with your company that were satisfactory, but others fell short.\"",
    "\"My feelings about my recent interaction with your company are fairly neutral.\"",
    "\"Overall, my experience with your company was neither positive nor negative.\"",
    "\"I don't have much to comment on regarding my recent purchase.\"",
    "\"The service provided was neither exceptional nor below average.\"",
    "\"My experience with your company was fairly standard.\"",
    "\"My feelings about my recent purchase are fairly ambivalent.\"",
    "\"There were aspects of my experience with your company that were satisfactory, but others could be improved.\"",
    "\"Overall, my interaction with your company was neither memorable nor forgettable.\"",
    "\"My experience with your product was neither remarkable nor disappointing.\"",
    "\"The service provided was adequate, but there's room for enhancement.\"",
    "\"I don't have any strong opinions about my recent interaction with your company.\"",
    "\"My feelings about my recent purchase are fairly neutral.\"",
    "\"Overall, my experience with your company was fairly average.\"",
    "\"I have no strong feelings about my recent interaction with your company.\"",
    "\"My experience with your product was neither outstanding nor subpar.\"",
    "\"The service provided met my basic needs but didn't go above and beyond.\"",
    "\"I don't have much to say about my recent purchase experience.\"",
    "\"My feelings about my recent interaction with your company are fairly ambivalent.\"",
    "\"Overall, my experience with your product was fairly average.\"",
    "\"The service provided was satisfactory, but there's room for improvement.\"",
    "\"My experience with your company was neither exceptional nor terrible.\"",
    "\"I don't have any strong opinions about my recent purchase.\"",
    "\"My feelings about my recent interaction with your company are fairly neutral.\"",
    "\"Overall, my experience with your company was fairly standard.\"",
    "\"The service provided was neither outstanding nor below average.\"",
    "\"I have no strong feelings about my recent purchase experience.\"",
    "\"My experience with your company was neither remarkable nor disappointing.\"",
    "\"I don't have much to comment on regarding my recent interaction with your company.\""
]

# Initialize lists to store results
sentiment_scores = []
sentiment_labels = []
transcripts_list = []

# Initialize SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Loop through transcripts and analyze sentiment for each one
for transcript in transcripts:
    # Split transcript using " symbol
    cleaned_transcript = transcript.strip("\"")

    # Perform sentiment analysis
    scores = sid.polarity_scores(cleaned_transcript)
    sentiment_scores.append(scores)

    # Determine sentiment label
    if scores['compound'] >= 0.1:
        sentiment_labels.append('positive')
    elif scores['compound'] <= -0.1:
        sentiment_labels.append('negative')
    else:
        sentiment_labels.append('neutral')

    # Store the transcript
    transcripts_list.append(cleaned_transcript)
print("Total transcripts processed:", len(transcripts_list))

# Create DataFrame
df = pd.DataFrame({'content': transcripts_list, 'sentiment_label': sentiment_labels})
df = pd.concat([df, pd.DataFrame(sentiment_scores)], axis=1)

# Print the DataFrame to check if the sentiment scores are added correctly
print("DataFrame:")
print(df)

# Streamlit Dashboard
st.title("Transcript Based Sentiment Analysis using NLTK")
st.image("https://news.itmo.ru/images/news/big/p9806.jpg", use_column_width=True)
st.markdown(
    """
    ## Welcome to the Transcript Based Sentiment Analysis System Dashboard! 
    
    Explore sentiment distribution, sentiment ratio, and more.
    """
)

# Navigation links
st.sidebar.title("EXPLORE")
selected_page = st.sidebar.radio("", ["Home", "Sentiment Ratios", "Sentiment Score", "Sentiments Distribution", "Sentiment Breakdown"])

if selected_page == "Sentiment Ratios":
    # Calculate sentiment ratio
    sentiment_ratio = df['sentiment_label'].value_counts(normalize=True).to_dict()
    for key in ['negative', 'neutral', 'positive']:
        if key not in sentiment_ratio:
            sentiment_ratio[key] = 0.0

    # Display sentiment ratio
    st.subheader("Sentiment Ratio")
    st.write(sentiment_ratio)

elif selected_page == "Sentiment Score":
    # Calculate sentiment score
    sentiment_ratio = df['sentiment_label'].value_counts(normalize=True).to_dict()
    for key in ['negative', 'neutral', 'positive']:
        if key not in sentiment_ratio:
            sentiment_ratio[key] = 0.0
    sentiment_score = (sentiment_ratio['positive'] * 0.5 + sentiment_ratio['neutral'] * 0.1 - sentiment_ratio['negative'] * 0.5)

    # Display sentiment score
    st.subheader("Sentiment Score")
    st.write(sentiment_score)

elif selected_page == "Sentiments Distribution":
    # Calculate sentiment ratio
    sentiment_ratio = df['sentiment_label'].value_counts(normalize=True).to_dict()
    for key in ['negative', 'neutral', 'positive']:
        if key not in sentiment_ratio:
            sentiment_ratio[key] = 0.0

    # Display sentiment pie chart
    st.subheader("Sentiments Distribution")
    fig_pie = px.pie(values=list(sentiment_ratio.values()), names=list(sentiment_ratio.keys()), title='Sentiments')
    st.plotly_chart(fig_pie)

elif selected_page == "Sentiment Breakdown":
    # Display sentiment breakdown
    st.subheader("Sentiment Breakdown")
    fig_scatter = px.scatter(df, y='sentiment_label', color='sentiment_label',
                             hover_data=['content'],
                             color_discrete_map={"negative": "firebrick", "neutral": "navajowhite", "positive": "darkgreen"})
    st.plotly_chart(fig_scatter)

#!/usr/bin/env python
# coding: utf-8

# #     EAFC24 Data Analytics EDA Project     #

# In[1]:


# IMPORTING PACKAGES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings 
warnings.filterwarnings ('ignore')


# #    Importing & Checking the Cleanliness of the Data

# In[2]:


# IMPORTING CSV FILES

df = pd.read_csv("C:/Anaconda/PROJECTS/EAFC Data/all_players.csv")
df.info()


# In[3]:


df.count()


# In[4]:


df = df.drop(['GK'], axis = 1)


# In[5]:


df.count()


# In[6]:


df.info()


# In[7]:


df.head()


# In[8]:


df.describe()


# #     Distribution of Player Age

# In[9]:


age_counts = df.groupby('Age')['Name'].count()
plt.figure(figsize=(12,8))
sns.barplot(x = age_counts.index, y = age_counts.values)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show


# #     Distribution of Overall Rating

# In[10]:


# Create a histplot for the 'Overall' column
position = df['Overall'].value_counts

# Create a Seaborn histplot with specified bins
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='Overall', kde=True, bins=45)  # Adjust the number of bins as needed

x_ticks = range(47, 92, 1)  # Adjust the range and step as needed
plt.xticks(x_ticks)

plt.title('Distribution of Overall Rating')
plt.xlabel('Overall')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# #     Distribution of Preferred Positions

# In[11]:


position_counts = df['Position'].value_counts(ascending=False)

# Create a Seaborn barplot 
plt.figure(figsize=(12, 8))
sns.barplot(x = position_counts.index, y = position_counts.values)  # Adjust the number of bins as needed

plt.title('Preferred Position Distribution')
plt.xlabel('Position')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# #     Top 20 Nationalities

# In[12]:


# TOP 20 NATIONALITIES

nationality_counts = df['Nation'].value_counts(ascending=False).head(20)

# Create a Seaborn barplot 
plt.figure(figsize=(12, 8))
sns.barplot(x = nationality_counts.index, y = nationality_counts.values)  # Adjust the number of bins as needed

plt.title('Top 20 Nationalities')
plt.xlabel('Nationality')
plt.xticks(rotation = 45, ha = 'right')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# #     Average Rating Per Position

# In[13]:


# AVERAGE RATING PER POSITION

position_rating =df.groupby('Position')['Overall'].mean()
plt.figure(figsize=(12,8))
ax=sns.barplot(x = position_rating.index, y = position_rating.values)
plt.title('Average Rating per Position')
plt.xlabel('Position')
plt.xticks(rotation = 45, ha = 'right')
plt.ylabel('Overall Rating')

for p in ax.patches:
    height = p.get_height()
    rounded_percent = round(height)  # Round to the nearest whole number
    ax.annotate(f'{rounded_percent}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', fontsize=9, color='black')

plt.tight_layout()
plt.show()


# #     How Does Skill Affect Overall Rating? 
# 
# #     Finding The Correlation Between Skill & Rating

# In[14]:


skill_columns = ['Shooting', 'Passing', 'Dribbling', 'Defending']
plt.figure(figsize=(12,6))

for skill in skill_columns:
    sns.histplot(df[skill], kde=True, label=skill)

plt.title('Distribution of Skill Ratings')
plt.xlabel('Skill Rating')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[15]:


skill_columns = ['Shooting', 'Passing', 'Dribbling', 'Defending']
overall_rating = df['Overall']

#Create scatter plots for each skill rating against the overall player rating
plt.figure(figsize=(24,6))

for i, skill in enumerate(skill_columns, 1):
    plt.subplot(1, len(skill_columns), i)
    sns.regplot(x=df[skill], y=overall_rating, line_kws={"color": "red"})  # Use regplot for linear regression
    plt.title(f'{skill} vs Overall Rating')
    plt.xlabel(skill)
    plt.ylabel('Overall Rating')
    
plt.tight_layout()
plt.show()


# #      Correlation Between Player Attributes

# In[16]:


# Select relevant attributes for correlation analysis
attributes = df[['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Acceleration', 'Sprint', 'Physicality']]

# Calculate the correlation matrix
correlation_matrix = attributes.corr()

# Plot the heatmap to visualize the correlations
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Between Player Attributes')
plt.show()


# #     Right or Left? - Preferred Foot Analysis

# In[17]:


foot_count = df['Preferred foot'].value_counts()

colors = ['#FA8072', '#4682b4']

explode = (0.1,0)

plt.figure(figsize = (8,8))
plt.pie(foot_count, labels=foot_count.index, autopct = '%1.1f%%', colors=colors, explode=explode, shadow=True)
plt.title('Preferred Foot Distribution')
plt.show()


# #     Gender Distribution

# In[18]:


gender_counts = df['Gender'].value_counts()

colors = ['#FA8072', '#4682b4']

explode = (0.2,0)

plt.figure(figsize=(8,8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=colors, explode=explode, shadow=True)
plt.title('Gender Distribution')
plt.show()


# #     Working Hard or Hardly Working - Distribution of Work Rates

# In[19]:


skill_columns = ['Att work rate', 'Def work rate']
plt.figure(figsize=(12,6))

for skill in skill_columns:
    rate_count = df[skill].value_counts()
    sns.histplot(df[skill], kde=True, label=skill)
    
plt.title('Distribution of Attacking and Defensive Work Rates')
plt.xlabel('Work Rate')
plt.ylabel('Frequency')
plt.legend()
plt.show


# #     Count of Players by Attacking/Defensive Work Rate

# In[20]:


# Define the order of the x-axis values
order = ['High', 'Medium', 'Low']

# Create the bar plot with the specified order
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Att work rate', palette="Set2", order=order)

# Set labels and title
plt.xlabel('Attacking Work Rate')
plt.ylabel('Count')
plt.title('Count of Players by Attacking Work Rate')

# COUNT OF PLAYERS BY DEFENSIVE WORK RATE

# Create the bar plot with the specified order
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Def work rate', palette="Set2", order=order)

# Set labels and title
plt.xlabel('Defensive Work Rate')
plt.ylabel('Count')
plt.title('Count of Players by Defensive Work Rate')

# Show the plot
plt.show()


# #      Distribution of Overall Work Rate Combinations

# In[21]:


# Combine 'Att work rate' and 'Def work rate' into 'Work Rate'
df['Work Rate'] = df['Att work rate'] + ' - ' + df['Def work rate']

# Create an interactive count plot using Plotly Express
fig = px.histogram(df, x='Work Rate', color='Gender', barmode='group',
                   title='Count of Work Rates by Gender', labels={'Work Rate': 'Work Rate'})

# Show the interactive plot
fig.show()


# #      Who Has The Moves? - Distribution of Skill Moves

# In[22]:


# WHO HAS THE MOVES? -- SKILL MOVE DISTRIBUTION 

plt.figure(figsize=(12,6))
plot = sns.countplot(data = df, x='Skill moves', palette='Set1')
plt.title('Count of Skill Moves')
plt.xlabel('Skill Moves')
plt.ylabel('Count')

for p in plot.patches:
    plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color='black', xytext=(0,5), textcoords='offset points')
                  
plt.show


# #     Top 50 Defenders by Defending/Physicality

# In[23]:


positions = ['CB', 'RB', 'LB', 'RWB', 'LWB', 'CDM']

#SCATTERPLOT
top_50_defenders = df.nlargest(50, ['Defending', 'Physicality'])

fig = px.scatter(top_50_defenders, x='Defending', y='Physicality', text='Name', title='Top 50 Players with Highest Defending and Physicality', color='Overall')

fig.update_traces(marker=dict(size=20))

fig.show()

#BARGRAPHs
top_50_players_by_defending = df[df['Position'].isin(positions)].nlargest(50, 'Defending')
top_50_players_by_physicality = df[df['Position'].isin(positions)].nlargest(50, 'Physicality')

fig_defending = px.bar(
    top_50_players_by_defending,
    x='Defending',
    y='Overall',
    title='Top 50 Defenders by Defense Rating',
    color = 'Overall',
    labels={'Defending', 'Defending'},
    hover_data=['Name', 'Club', 'Nation', 'Position'],
    hover_name='Name')

fig_physicality = px.bar(
    top_50_players_by_physicality,
    x='Physicality',
    y='Overall',
    title='Top 50 Defenders by Physicality Rating',
    color = 'Overall',
    labels={'Physicality', 'Physicality'},
    hover_data=['Name', 'Club', 'Nation', 'Position'],
    hover_name='Name')

fig_defending.show()
fig_physicality.show()


# #     Which Positions Have the Highest Attributes?

# In[24]:


attributes = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physicality']

average_values = df.groupby('Position')[attributes].mean().reset_index()

average_values[attributes] = average_values[attributes].apply(lambda x: round(x,2))

average_values


# #    The Best in the World - Comparing the Highest Rated Players per Position

# In[25]:


#COMPARING THE BEST IN THE WORLD - FORWARDS

# Define the radar attributes and positions of interest
radar_attributes = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality"]
positions_of_interest = ['ST', 'RW', 'LW', 'CF']

# Filter the DataFrame for players with positions in the list
filtered_players = df[df['Position'].isin(positions_of_interest)]

# Calculate the average attributes for players in the specified positions
average_attributes = filtered_players[radar_attributes].mean()

# Sort the DataFrame to get the top 10 forwards by 'Overall'
top_10_forwards = filtered_players.nlargest(10, 'Overall')

# Create a subplot with two radar charts
fig = go.Figure()

# Create the first radar chart for average attributes
fig.add_trace(go.Scatterpolar(
    r=average_attributes.values,
    theta=average_attributes.index,
    fill='toself',
    name="Average Attributes"
))

# Create the second radar chart for the top 10 forwards by 'Overall'
for i, player in top_10_forwards.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=player[radar_attributes].values,
        theta=radar_attributes,
        fill='toself',
        name=player['Name'],
        visible='legendonly' if i >= 10 else True  # Only show legend for the top 10
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    title='Comparison of Top 10 Forwards',
    showlegend=True
)

# Add labels for each data point
for cat, val in zip(average_attributes.index, average_attributes.values):
    fig.add_annotation(
        text=f'Avg {cat}: {val:.2f}',
        x=cat,
        y=val,
        showarrow=False
    )

fig.show()


# In[26]:


#COMPARING THE BEST IN THE WORLD - MIDFIELDERS

# Define the radar attributes and positions of interest
radar_attributes = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality"]
positions_of_interest = ['CM', 'RM', 'LM', 'CAM','CDM']

# Filter the DataFrame for players with positions in the list
filtered_players = df[df['Position'].isin(positions_of_interest)]

# Calculate the average attributes for players in the specified positions
average_attributes = filtered_players[radar_attributes].mean()

# Sort the DataFrame to get the top 10 forwards by 'Overall'
top_10_midfielders = filtered_players.nlargest(10, 'Overall')

# Create a subplot with two radar charts
fig = go.Figure()

# Create the first radar chart for average attributes
fig.add_trace(go.Scatterpolar(
    r=average_attributes.values,
    theta=average_attributes.index,
    fill='toself',
    name="Average Attributes"
))

# Create the second radar chart for the top 10 forwards by 'Overall'
for i, player in top_10_midfielders.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=player[radar_attributes].values,
        theta=radar_attributes,
        fill='toself',
        name=player['Name'],
        visible='legendonly' if i >= 10 else True  # Only show legend for the top 10
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    title='Comparison of Top 10 Midfielders',
    showlegend=True
)

# Add labels for each data point
for cat, val in zip(average_attributes.index, average_attributes.values):
    fig.add_annotation(
        text=f'Avg {cat}: {val:.2f}',
        x=cat,
        y=val,
        showarrow=False
    )

fig.show()


# In[27]:


#COMPARING THE BEST IN THE WORLD - DEFENDERS

# Define the radar attributes and positions of interest
radar_attributes = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality"]
positions_of_interest = ['CB', 'LB', 'RB', 'LWB','RWB']

# Filter the DataFrame for players with positions in the list
filtered_players = df[df['Position'].isin(positions_of_interest)]

# Calculate the average attributes for players in the specified positions
average_attributes = filtered_players[radar_attributes].mean()

# Sort the DataFrame to get the top 10 forwards by 'Overall'
top_10_defenders = filtered_players.nlargest(10, 'Overall')

# Create a subplot with two radar charts
fig = go.Figure()

# Create the first radar chart for average attributes
fig.add_trace(go.Scatterpolar(
    r=average_attributes.values,
    theta=average_attributes.index,
    fill='toself',
    name="Average Attributes"
))

# Create the second radar chart for the top 10 forwards by 'Overall'
for i, player in top_10_defenders.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=player[radar_attributes].values,
        theta=radar_attributes,
        fill='toself',
        name=player['Name'],
        visible='legendonly' if i >= 10 else True  # Only show legend for the top 10
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    title='Comparison of Top 10 Defenders',
    showlegend=True
)

# Add labels for each data point
for cat, val in zip(average_attributes.index, average_attributes.values):
    fig.add_annotation(
        text=f'Avg {cat}: {val:.2f}',
        x=cat,
        y=val,
        showarrow=False
    )

fig.show()


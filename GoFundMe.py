#!/usr/bin/env python
# coding: utf-8

# # Project 2: GoFundMe Campaigns
# 
# 
# 
# ## Due Thursday, March 11 at 11:59pm
# 
# 
# <img src="data/gofundme.png" width=60%>
# 
# 
# Welcome to the Final Project, where we'll be exploring data from GoFundMe campaigns!

# ## Logistics
# 
# **Deadline.** This assignment is due Thursday, March 11 at 11:59pm. You are given six slip days throughout the quarter to extend deadlines. See the syllabus for more details. With the exception of using slip days, late work will not be accepted unless you have made special arrangements with your instructor.
# 
# **Partners.** You may **work in pairs** for this assignment, and you are encouraged to do so! If you work in a pair, you must work with someone from your team, and you should submit one notebook to Gradescope for the both of you, while designating your partner so that both of you receive credit.
# 
# **Rules.** Don't share your code with anybody but your partner. You are welcome to discuss questions with other students, but don't share the answers. The experience of solving the problems in this project will prepare you for the final exam and your future in data science. If someone asks you for the answer, resist! Instead, you can demonstrate how you would solve a similar problem.
# 
# **Support.** You are not alone! Come to office hours, post on Piazza, and talk to your classmates. If you want to ask about the details of your solution to a problem, make a private Piazza post and the staff will respond. All of the concepts necessary for this project are found in the textbook or supplemental textbook. If you are stuck on a particular problem, reading through the relevant textbook section often will help clarify the concept.
# 
# **Tests.** Passing the tests for a question **does not** mean that you answered the question correctly. Tests usually only check that your table has the correct column labels or that your answer is of the correct type. However, more tests will be applied to verify the correctness of your submission in order to assign your final score, so be careful and check your work!
# 
# **Advice.** First, **start early**. As you may know from the Midterm Project, projects are complex and time-consuming. Second, develop your answers incrementally. To perform a complicated task, break it up into steps, perform each step on a different line, give a new name to each result, and check that each intermediate result is what you expect. You can add any additional names or functions you want to the provided cells, and you can add additional cells as needed. Don't try to do everything in one cell without seeing the intermediate output. In particular, for simulations where you need to do something many times, first just do the process once and make sure the results look reasonable. Then wrap your code inside a for loop to repeat it. Similarly, for defining functions, first write code that will produce the desired output for a single fixed input. Then, once you know it's working, you can put that code inside a function and change the input to be a variable. 
# 
# **Long Simulations.** If any of your cells are taking more than five minutes to run, you are probably doing something wrong. You can sometimes speed things up by making sure you have a table of only the rows and columns you need to do your analysis, which should be defined outside the for loop of your simulation. Make sure your table is as small as possible in both rows and columns. When possible, try to avoid using additional for loops and queries inside a simulation, and see if a faster method, like a numpy method or groupby, could be used instead. 
# 
# Let's get started!

# ## Background
# 
# GoFundMe is an online crowdfunding platform where people can raise funds for personal, business and charitable work. GoFundMe enables users to create their own fundraising website, where they can describe their fundraising cause, upload photos and videos, and set a goal for how much they want to raise. People can also share the fundraiser through social networks like Facebook, Twitter, and email. 
# 
# GoFundMe is currently the largest crowdfunding platform ever, and in one decade, has raised over 9 billion dollars for various causes, with contributions coming from over 120 million donors ([Source: Wikipedia](https://en.wikipedia.org/wiki/GoFundMe)).
# 
# Fun Fact: GoFundMe was founded in San Diego, CA!
# 
# In this project, we will be analyzing a dataset of fundraisers that users have posted on GoFundMe. This data was scraped from the GoFundMe main page, category pages, and campaign URLs. The data was gathered in early 2021, pulling the most recent fundraisers. 
# 
# This dataset was compiled by UCSD students Gauri Samith, Derek Leung, Emily Chen, and Vincent Lee for use in a personal project, organized through the Data Science Student Society’s Projects Committee, and they were kind enough to share it with us for this DSC 10 project. If you want to see the analysis they've done on this dataset and see some other great data science projects, check out the [DS3 Project Showcase](data/proj_showcase.png), on Monday, February 22, 2021 from 3-5pm. 

# In[1]:


# please don't change this cell, but do make sure to run it
import babypandas as bpd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np

import otter
import numbers
import IPython
grader = otter.Notebook()


# The data is in a CSV called `gfm_data.csv`. We will read this file into a dataframe named `gfm_data`.

# In[2]:


gfm_data = bpd.read_csv('data/gfm_data.csv')
gfm_data


# Here, each row represents a different fundraiser, also called a campaign. 
# 
# There are nine columns of data, reading from left to right:
# 
# 1. `Category`: Describes the type of fundraiser. There are 14 different categories of fundraiser represented in our dataset: 'Animals', 'Business', 'Community', 'Competition', 'Creative', 'Emergency', 'Event', 'Faith', 'Family', 'Medical', 'Memorial', 'Newlywed', 'Sports', and 'Wishes'.
# 
# 2. `Title`: The title of the fundraiser, which describes the fundraiser in a few words.
# 
# 3. `Location`: The location where the fundraiser takes place, stored as "city, state", e.g. "San Diego, CA".
# 
# 4. `Amount_Raised`: The total amount of money donated, measured in dollars.
# 
# 5. `Goal`: The total amount of money that the fundraiser originally hopes to receive (measured in dollars).
# 
# 6. `Days_of_Fundraising`: The number of days for which the fundraiser has been actively soliciting donations.
# 
# 7. `Number_of_Donors`: The number of people who donated to the fundraiser. 
# 
# 8. `FB_Shares`: The number of times the fundraiser has been shared through Facebook. 
# 
# 9. `Text`: The text description that accompanies the fundraiser, usually describing why funds are being raised.
# 
# In the `Number_of_Donors` column and the `FB_Shares` column, entries are stored as strings, and large numbers are abbreviated by using "K" to indicate thousands. For example, an entry `3K` would correspond to 3 * 1000 = 3000.
# 
# Note that much of the data for each fundraiser is determined by the user who creates the campaign. When creating a campaign, the user selects an appropriate category and title, chooses the location, sets the fundraising goal, and adds the text description. 

# ## Outline of the Project 
# 
# The project is divided into six sections. The outline below includes links that will take you directly to each section. The number of questions in each section is also listed here. 
# 
# Within each section, work through the questions in order. 
# 
# Section 1 must be completed first. After that, you can work through the other sections in any order, though Section 3 must be done before you can do Section 4. The sections are ordered by when we learned the relevant topics in class. As of the release of this project, you have already learned enough to do sections 1 through 5. The content of section 6 is regression, which will be covered in the last two weeks of the quarter. 
# 
# Sections 3 and 4 are the most challenging sections.
#   
# -  Section 1. [Data Cleaning](#section1)  
#      - 7 questions
#      - In this section, we'll prepare the data for our analysis.
# -  Section 2. [Popular Fundraising Categories](#section2) 
#      - 8 questions
#      - In this section, we'll explore whether certain fundraising categories are more popular than others.
# -  Section 3. [Medical vs. Creative](#section3)  
#      - 11 questions
#      - In this section, we'll compare the fundraising outcomes for Medical campaigns to the outcomes for Creative campaigns.
# -  Section 4. [The Power of Words](#section4)  
#      - 7 questions
#      - In this section, we'll see whether the length of the text description and the actual words used in the description have an impact on the amount of money raised.
# -  Section 5. [Average Amount Raised](#section5)  
#      - 3 questions
#      - In this section, we'll estimate the average amount of money raised by all GoFundMe campaigns.
# -  Section 6. [Sharing the Wealth](#section6) 
#      - 10 questions
#      - In this section, we'll predict how much money a campaign brings in based on the number of Facebook shares it's received.

# <a id='section1'></a>
# ## 1. Data Cleaning

# As a data scientist, data cleaning is where you will spend a significant portion of your time. This project is no different!
# 
# Let's start by examining the `Number_of_Donors` and `FB_Shares` columns. The values in these columns are strings, not integers, so we can't actually do things like compare values numerically. Moreover, some values are in thousands, such as '72.5K', and others are fewer than 1000, such as '366'. Let's get this data in a more usable format.
# 
# **Question 1.1.** Define a function named `convert_units` which converts a string representation of a number to an integer. This function should work whether or not the string representation has a ‘K’ in it (representing thousands).
# 
# *Hint*: Use a [string method](https://docs.python.org/3/library/stdtypes.html#string-methods) to deal with the 'K'.

# In[ ]:





# In[3]:


def convert_units(string_rep):
    k_tf= string_rep.endswith('K')
    if k_tf == True:
        string_rep_nok= string_rep.strip('K')
        string_rep_float= float(string_rep_nok)
        string_rep_final_float = 1000* string_rep_float
        string_rep_final= int(string_rep_final_float)
    else:
        string_rep_final_float= float(string_rep)
        string_rep_final= int(string_rep_final_float)
    return string_rep_final


# In[4]:


grader.check("q1_1")


# In[ ]:





# **Question 1.2.** Overwrite the columns for `Number_of_Donors` and `FB_Shares` with columns of the same name, but containing integers instead of strings. The dataframe should still be named `gfm_data`.
# 
# *Note*: The names of all the columns **must** stay the same and the order of the columns **must** stay the same.

# In[5]:


gfm_data = gfm_data.assign(Number_of_Donors= gfm_data.get('Number_of_Donors').apply(convert_units))
gfm_data = gfm_data.assign(FB_Shares= gfm_data.get('FB_Shares').apply(convert_units))
gfm_data


# In[6]:


grader.check("q1_2")


# In[ ]:





# Next, let's add some columns derived from existing columns. 
# 
# **Question 1.3.** Create a new dataframe called `gfm_campaigns` that has the same columns as `gfm_data`, plus two more:
# 
# 1. `Proportion_Raised`, which has the proportion of the overall goal that the campaign raised. This proportion should be rounded (not truncated) to **3** decimal places. 
# 2. `Average_Donation_Amount`, which has the average donation amount per donor. Since this is a dollar amount, it should be rounded (not truncated) to **2** decimal places.

# In[7]:


def round3(num):
    return round(num,3)
def round2(num):
    return round(num,2)
gfm_campaigns = gfm_data.assign(Proportion_Raised= ((gfm_data.get('Amount_Raised')/(gfm_data.get('Goal')))).apply(round3))
gfm_campaigns = gfm_campaigns.assign(Average_Donation_Amount= ((gfm_campaigns.get('Amount_Raised'))/(gfm_campaigns.get('Number_of_Donors'))).apply(round2))
gfm_campaigns


# In[8]:


grader.check("q1_3")


# In[ ]:





# **Question 1.4.** Create a new dataframe called `gfm_success`, which has all the information from the `gfm_campaigns` dataframe plus a new column titled `How_Successful` that indicates how successful a campaign was, depending on its `Proportion_Raised`. If x is the proportion raised, we use the table below to define how successful a campaign is.
# 
# | Range             | How_Successful |
# | ----------------- | ------------------ |
# | 0.0 $\leq$ x $\leq$ 0.20 | 'highly unsuccessful' |
# | 0.20 $<$ x $\leq$ 0.50 | 'moderately unsuccessful' |
# | 0.50 $<$ x $\leq$ 0.80 | 'moderately successful' |
# | 0.80 $<$ x $<$ 1.0 | 'highly successful' |
# | x $\geq$ 1.0 | 'extremely successful' |

# In[ ]:





# In[9]:


def sucess(prop):
    if (prop >= 0.0) & (prop <=0.2):
        return 'highly unsuccessful'
    if (prop > 0.2) & (prop <=0.5):
        return 'moderately unsuccessful'
    if (prop > 0.5) & (prop <=0.8):
        return 'moderately successful'
    if (prop > 0.8) & (prop < 1.0):
        return 'highly successful'
    if prop>= 1.0:
        return 'extremely successful'
gfm_success = gfm_campaigns.assign(How_Successful=gfm_campaigns.get('Proportion_Raised').apply(sucess))
gfm_success


# In[10]:


grader.check("q1_4")


# In[ ]:





# Notice that the `Text` column contains the description of the campaign, which visitors to the campaign page will see. As part of this project, we'd like to explore what makes a description successful in raising money. For example, do longer descriptions raise more money or is better to keep things brief? Are there certain keywords like "mother" or "father" associated with better fundraising outcomes? To answer these types of questions, we'll need to look inside the `Text` column. The data cleaning below will help us extract the information we need more easily.
# 
# **Question 1.5.** Create a new dataframe called `gfm`, which has all the information from the `gfm_success` dataframe plus a new column called `Num_Chars` with the length of the text description for each campaign, as an int.

# In[11]:


def length(strings):
    return len(strings)
gfm = gfm_success.assign(Num_Chars=gfm_success.get('Text').apply(length))
gfm


# In[ ]:





# In[12]:


grader.check("q1_5")


# In[ ]:





# **Question 1.6.** Overwrite the `Text` column in the `gfm` dataframe so that all the descriptions appear in lowercase.
# 
# *Hint:* Check out Python's built-in [string methods](https://docs.python.org/3/library/stdtypes.html#string-methods).

# In[13]:


def lowercase(strings):
    return strings.lower()
gfm = gfm.assign(Text= gfm.get('Text').apply(lowercase))
gfm


# In[14]:


grader.check("q1_6")


# Great! Now we have a clean dataset that we can use to start answering questions. For the rest of this project, use the `gfm` dataframe unless otherwise instructed.

# <a id='section2'></a>
# # 2. Popular Fundraising Categories

# On GoFundMe, people can fundraise for a variety of causes. In our `gfm` dataframe, each fundraiser has an associated category, such as "Memorial" or "Newlywed". Campaigns can come from a variety of categories, but are certain categories more prevalent than others? 
# 
# We'll think of the fundraisers in our `gfm` dataframe as a *sample* taken from a larger *population* of all GoFundMe campaigns. In this section, we'll explore whether our sample comes from a population where all fundraiser categories are equally likely, or if certain categories are more likely to occur than others. 
# 
# Especially given the circumstances of 2020 and early 2021, this can give us an understanding of the types of causes where people need financial support the most. 
# 
# Let's start by looking at the distribution of categories in our dataset.
# 
# **Question 2.1.** Find the total number of fundraisers in each category. Set the variable `category_count` to be a Series, indexed by category, containing the number of fundraisers per category. Order the values of `category_count` from least to greatest.

# In[15]:


category_count = gfm.groupby('Category').count().sort_values('Title').get('Title')
category_count


# In[16]:


grader.check("q2_1")


# In[ ]:





# If you did this question correctly, you should see that two categories, 'Animals' and 'Competition' are outliers compared to the rest. There are definitely fewer fundraisers in these categories than others, but for the other categories, the numbers are more similar. We're going to discard the  'Animals' and 'Competition' categories and just work with the other categories for this section of the project.
# 
# **Question 2.2.**  Create a new dataframe called `gfm_reduced`, which should be the same as `gfm` but with any fundraiser from the 'Animals' or 'Competition' category removed. Then for the **remaining** categories, create a new Series, indexed by category, called `category_proportion`. This should contain the proportion of fundraisers in `gfm_reduced` that come from each category. Order the values of `category_proportion` from least to greatest.

# In[ ]:





# In[17]:


gfm_reduced = gfm[(gfm.get('Category')!= 'Animals')&(gfm.get('Category')!= 'Competition')]
total_fund_num= gfm_reduced.shape[0]
category_proportion = gfm_reduced.groupby('Category').count().sort_values('Title').get('Title')/total_fund_num
category_proportion


# In[ ]:





# In[18]:


grader.check("q2_2")


# In[ ]:





# These numbers should all look pretty similar, meaning that among the remaining 12 categories, our sample of fundraisers has a similar amount of fundraisers in each category. Do we think that the fundraisers posted on GoFundMe are equally likely to come from any of these 12 categories, or do we think that certain categories are more common than others? In other words, the number of fundraisers in each category in our dataset is not *exactly* the same, but are the differences we see just due to chance, or do they reflect a fact about the population, that certain categories are more popular than others?
# 
# This is a good time to do a hypothesis test, because we want to compare our data (`gfm_reduced`) to a model (a uniform distribution). Here are our hypotheses:
# 
# - **Null Hypothesis:** The sample of fundraisers in `gfm_reduced` is drawn from a population of fundraisers in which each category is equally likely.
# 
# - **Alternative Hypothesis:** The sample of fundraisers in `gfm_reduced` is drawn from a population where certain fundraiser categories are more likely than others.

# We want to test these hypotheses by simulating data under the assumption of the null hypothesis to create simulated test statistics, then compare these simulated statistics to the observed value of the test statistic. We will use the total variation distance, which quantifies the distance between two distributions. 
# 
# **Question 2.3.** This choice of test statistic is not unique. Which of the following test statistics could we have used instead? There may be more than one correct answer.
# 
# 1. The number of fundraisers in the "Family" category.
# 2. The proportion of fundraisers in the "Family" category.
# 3. The smallest number of fundraisers in any of the 12 remaining categories.
# 4. The largest proportion of fundraisers in any of the 12 remaining categories.
# 5. The average proportion of fundraisers in each of the 12 remaining categories.
# 6. The sum of all proportions of fundraisers in each of the 12 remaining categories.
# 7. The sum of all signed differences between the observed and expected proportion of fundraisers in each of the 12 remaining categories.
# 8. The sum of all absolute differences between the observed and expected proportion of fundraisers in each of the 12 remaining categories.
# 
# When you're deciding on which test statistics could work here, remember that you need to be able to use the observed value of the test statistic to distinguish between the two hypotheses. 
# 
# Set `valid_test_stats` to a list of numbers 1 through 8 corresponding to **all** of the test statistics that would be suitable for this hypothesis test.

# In[19]:


valid_test_stats = [3,4,8]
valid_test_stats


# In[20]:


grader.check("q2_3")


# **Question 2.4.** As our test statistic, we will use total variation distance to the uniform distribution. What values of this test statistic support the alternative hypothesis?  Set `q2_4_answer` to your answer choice.
# 
# 1. High values only 
# 2. Low values only  
# 3. Both high and low values
# 4. Moderate values

# In[21]:


q2_4_answer = 1
q2_4_answer


# In[22]:


grader.check("q2_4")


# **Question 2.5.** Complete the definition of the function `calculate_tvd`, which takes as input two distributions (lists, arrays, or Series containing proportions in each category) and calculates the total variation distance between the two distributions. 

# In[23]:


def calculate_tvd(dist1, dist2): 
    difference= np.array(dist1) - np.array(dist2)
    abs_diff= np.abs(difference)
    sum_diff= sum(abs_diff)
    divided=  sum_diff/2
    return divided


# In[24]:


grader.check("q2_5")


# In[ ]:





# Now, we want to calculate the observed value of the test statistic for our sample. How different is the distribution of the data in `gfm_reduced` from the uniform distribution?
# 
# **Question 2.6.** Create a list called `null_proportion` that represents the distribution of fundraisers into categories according to the assumption of the null hypothesis. Then calculate the TVD between this distribution and our observed distribution of fundraisers into categories. Save the result as `observed_tvd`.
# 
# 
# *Hint*: For `null_proportion` there are 12 different categories, and the null hypothesis says that each category is equally likely.

# In[25]:


null_proportion = [1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12, 1/12,]
observed_tvd = calculate_tvd(category_proportion,null_proportion)
null_proportion , observed_tvd


# In[26]:


grader.check("q2_6")


# In[ ]:





# **Question 2.7.** Now we want to simulate, many times, how the test statistic might turn out if we draw a sample of fundraisers from a uniform distribution, as stated in the null hypothesis. Calculate 10,000 simulated values of the test statistic according to the assumptions of the null hypothesis. Store these statistics in an an array called `simulated_tvds`.
# 
# *Hint:* Remember, the size of the sample you create from the uniform distribution should be equal to the number of fundraisers in `gfm_reduced`, for consistency. 

# In[27]:


simulated_tvds = np.array([])
repetitions= 10000
sample_size= gfm_reduced.shape[0]
for i in np.arange(repetitions):
    simulation= np.random.multinomial(sample_size,null_proportion)
    simultation_prop= simulation/sample_size
    tvd_sim= calculate_tvd(simultation_prop, null_proportion)
    simulated_tvds= np.append(simulated_tvds,tvd_sim )
    
simulated_tvds


# In[28]:


grader.check("q2_7")


# In[ ]:





# Run the cell below to plot a histogram of simulated TVDs. The red line shows the value of your observed TVD.

# In[29]:


bpd.DataFrame().assign(TVD=simulated_tvds).plot(kind='hist', density=True, bins=20)
plt.axvline(observed_tvd, color='red')


# **Question 2.8.** Which is the best interpretation of the results of this hypothesis test?
# 1. The observed statistic is similar to many of the simulated statistics. This suggests that the null hypothesis is true.
# 2. The observed statistic is similar to many of the simulated statistics. This suggests that the alternative hypothesis is true.
# 3. The observed statistic is smaller than many of the simulated statistics. This suggests that the null hypothesis is true.
# 4. The observed statistic is smaller than many of the simulated statistics. This suggests that the alternative hypothesis is true. 
# 5. The observed statistic is larger than many of the simulated statistics. This suggests that the null hypothesis is true.
# 6. The observed statistic is larger than many of the simulated statistics. This suggests that the alternative hypothesis is true.
# 
# Store the number corresponding to your answer to the variable `test_interpretation` below.

# In[30]:


test_interpretation = 3
test_interpretation


# In[31]:


grader.check("q2_8")


# <a id='section3'></a>
# # 3. Medical vs. Creative

# Some GoFundMe campaigns aim to raise money to cover unexpected medical bills or to fund needed surgeries. Others are community campaigns to support creative ventures like theatre programs in schools. In this section, we'll explore whether fundraisers categorized as "Medical" tend to raise a different amount of money than fundraisers categorized as "Creative." More specifically, we will determine whether the `Amount_Raised` from these two categories follow different distributions, using an A/B test.

# **Question 3.1.** Create a table called `med_creative` that contains all of the Medical and Creative campaigns.

# In[32]:


med_creative = gfm[(gfm.get('Category')=='Creative') | (gfm.get('Category')=='Medical') ]
med_creative


# In[33]:


grader.check("q3_1")


# In[ ]:





# The overlaid histogram below allows us to compare the distributions of the total amount raised by Medical campaigns and Creative campaigns.

# In[34]:


# Don't change this cell, just run it.
(
    med_creative[med_creative.get('Category') == 'Medical']
    .get('Amount_Raised')
    .plot(kind='hist', label='Medical', color='green', alpha = 0.6, bins = 25, density = True)
)
(
    med_creative[med_creative.get('Category') == 'Creative']
    .get('Amount_Raised').plot(kind='hist', label='Creative', color='red', alpha = 0.5, bins = 25, density = True)
)
plt.xlabel('Amount Raised')

# Scaling the graph down to better visualize the comparison
scale_factor = 0.4

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor);
plt.ylim(ymin * scale_factor, ymax * scale_factor);
plt.legend(['Medical', 'Creative'])


# Notice that both distributions are strongly skewed right, which should make sense because there is no upper bound on the total amount raised, but there is a lower bound ($0). And it's pretty hard to raise a huge amount of money!
# 
# It should be fairly obvious from this histogram the distribution of total amount raised is substantially different for Medical and Creative campaigns. It seems like Medical campaigns are a lot more lucrative, that is, they raised more money.
# 
# Are these differences significant or are they simply due to chance? You can probably guess the right answer in this case, but let's do a permutation test to make sure:
# 
# **Null Hypothesis**: In the population of GoFundMe campaigns, the distribution of total amount raised is the same for campaigns categorized as 'Medical' and 'Creative'.
# 
# **Alternative Hypothesis**: In the population of GoFundMe campaigns, 'Medical' campaigns raise more money on average than 'Creative' campaigns.
# 
# In order to test our hypotheses, we will use as our test statistic the difference between the mean amount raised by Medical campaigns and the mean amount raised by Creative campaigns (Medical minus Creative).
# 
# First, we'll create a few general functions that will help us complete this permutation test and others like it. Remember our advice from earlier:
# 
# Don't try to do everything in one cell without seeing the intermediate output. In particular, for simulations where you need to do something many times, first just do the process once and make sure the results look reasonable. Then wrap your code inside a for loop to repeat it. Similarly, for defining functions, first write code that will produce the desired output for a single fixed input. Then, once you know it's working, you can put that code inside a function and change the input to be a variable. 

# **Question 3.2.** Create a function named `difference_of_means` which takes four inputs:
# 1. `table`: a table with a column called "Category"
# 2. `first_category`: a string representing one possible value in the "Category" column
# 3. `second_category`: a string representing another possible value in the "Category" column
# 4. `comparison_column`: the label of a column in `table` whose values we want to compare across categories
# 
# The function should return the the difference in means of the data in `comparison_column` between the two given categories (first minus second). 

# In[35]:


def difference_of_means(table, first_category, second_category, comparison_column):
    new_table_first= table[table.get('Category')==first_category]
    new_table_second= table[table.get('Category')==second_category]
    mean_first= new_table_first.get(comparison_column).mean()
    mean_second= new_table_second.get(comparison_column).mean()
    return mean_first- mean_second


# In[36]:


grader.check("q3_2")


# In[ ]:





# **Question 3.3.** Create a function named `run_permutation_test` which takes four inputs:
# 1. `table`: a table with a column called "Category"
# 2. `first_category`: a string representing one possible value in the "Category" column
# 3. `second_category`: a string representing another possible value in the "Category" column
# 4. `comparison_column`: the label of a column in `table` whose values we want to compare across categories
# 5. `n`: the number of simulations
# 
# The function should return an array containing `n` simulated values of the difference in mean between the two categories. 
# 
# *Hint:* You should call the function `difference of means` within `run_permutation_test.`

# In[37]:


def run_permutation_test(table, first_category, second_category, comparison_column, n):
    differences= np.array([])
    for i in np.arange(n):
        new_col= np.random.permutation(table.get(comparison_column))
        shuf_table= table.assign(shuf_comparison_column= new_col)
        dif= difference_of_means(shuf_table, first_category, second_category, 'shuf_comparison_column')
        differences= np.append(differences, dif)
    return differences


# In[38]:


grader.check("q3_3")


# In[ ]:





# Now, let's return to the permutation test we wanted to perform. Recall the hypotheses:
# 
# **Null Hypothesis**: In the population of GoFundMe campaigns, the distribution of total amount raised is the same for campaigns categorized as 'Medical' and 'Creative'.
# 
# **Alternative Hypothesis**: In the population of GoFundMe campaigns, 'Medical' campaigns raise more money on average than 'Creative' campaigns.
# 
# **Question 3.4.** Use a function you've defined to assign `observed_mean_difference` to the observed value of the test statistic for this permutation test (Medical minus Creative).

# In[39]:


observed_mean_difference = difference_of_means(med_creative, 'Medical', 'Creative', 'Amount_Raised')
observed_mean_difference


# In[40]:


grader.check("q3_4")


# **Question 3.5.** Use a function you've defined to assign `mean_differences` to an array of one thousand simulated values of the test statistic for this permutation test.

# In[41]:


mean_differences = run_permutation_test(med_creative, 'Medical', 'Creative', 'Amount_Raised',1000)


# In[ ]:





# In[42]:


grader.check("q3_5")


# In[ ]:





# The next cell plots a histogram of the test statistics stored in `mean_differences` as well as a red line for `observed_mean_difference`.

# In[43]:


# Don't change this cell, just run it.
bpd.DataFrame().assign(DifferenceInMeans=mean_differences).plot(kind='hist')
plt.axvline(observed_mean_difference, color='red')


# **Question 3.6.** Assign `p_val` to the p-value from the permutation test above.

# In[44]:


p_val = (observed_mean_difference<=mean_differences).mean()
p_val


# In[45]:


grader.check("q3_6")


# As expected based on our initial overlaid histogram, we'll reject the null hypothesis and conclude that Medical campaigns do raise significantly more money than Creative campaigns. 
# 
# This example can be used to illustrate how finding a statistically significant association between two variables (category and amount raised, in this instance) does not necessarily prove a causal relationship. If we re-categorized a Creative fundraiser as a Medical fundraiser, it probably wouldn't earn more money just because of its new classification. 
# 
# Nonetheless, we have discovered an interesting fact: Medical campaigns do raise more money than Creative campaigns! We could compare other pairs of categories, too, and in fact, it should be quite easy to do so, now that we have written some general functions. Feel free to try it out below!

# In[46]:


# Pick a different pair of categories, and see if there is a significant difference in the amount of money raised. 
run_permutation_test(gfm,'Animals', 'Creative', 'FB_Shares', 100)


# Earlier in this project, you created a column called `Average_Donation_Amount`, which took into account the total amount raised as well as the number of donors. This provides us another perspective on the success of a GoFundMe campaign. Continuing our examination of Medical and Creative campaigns, let's run a different permutation test with the goal of determining whether campaigns from these two categories follow different distributions for the average donation made.
# 
# The overlaid histogram below allows us to compare the distributions of the average donation amount for Medical campaigns and Creative campaigns.

# In[47]:


# Don't change this cell, just run it.
(
    med_creative[med_creative.get('Category') == 'Medical']
    .get('Average_Donation_Amount')
    .plot(kind='hist', label='Medical', color='green', alpha = 0.6, bins = 25, density = True)
)
(
    med_creative[med_creative.get('Category') == 'Creative']
    .get('Average_Donation_Amount').plot(kind='hist', label='Creative', color='red', alpha = 0.5, bins = 25, density = True)
)
plt.xlabel('Amount Raised')

# Scaling the graph down to better visualize the comparison
scale_factor = 0.4

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor);
plt.ylim(ymin * scale_factor, ymax * scale_factor);
plt.legend(['Medical', 'Creative'])


# We can use the shape of these distributions to tell us something about which category had a greater average donation amount. The red distribution has a strong right skew, which will increase the mean of the data, whereas the green distribution is only slightly right skewed.  This means that the red distribution, representing the Creative campaigns, from our sample has a higher average donation amount than the green distribution, representing the Medical campaigns. In our sample of fundraisers, donors give more money, on average, to Creative campaigns than they do to Medical campaigns. Does this reflect a truth about the larger population of all GoFundMe campaigns, or is this just due to chance?
# 
# Again, a permutation test can help us answer this question. Here are our hypotheses:
# 
# **Null Hypothesis**: In the population of GoFundMe campaigns, the distribution of average amount donated by each donor is the same for campaigns categorized as 'Medical' and 'Creative'.
# 
# **Alternative Hypothesis**: In the population of GoFundMe campaigns, 'Creative' campaign donors give more money, on average, than donors to 'Medical' campaigns.
# 
# In order to test our hypotheses, we will use as our test statistic the difference between the means of the average donation amount made by donors of Creative and Medical campaigns (Creative minus Medical).
# 
# **Question 3.7.** Use a function you've defined to assign `observed_mean_difference_2` to the observed value of the test statistic for this permutation test (Creative minus Medical).
# 
# *Hint*: Be careful, this is Creative minus Medical, whereas the last test was Medical minus Creative.

# In[48]:


observed_mean_difference_2 = difference_of_means(med_creative, 'Creative','Medical','Average_Donation_Amount' )
observed_mean_difference_2


# In[49]:


grader.check("q3_7")


# **Question 3.8.** Use a function you've defined to assign `mean_differences_2` to an array of one thousand simulated values of the test statistic for this permutation test.

# In[50]:


mean_differences_2 = run_permutation_test(med_creative, 'Creative','Medical','Average_Donation_Amount' , 1000)
mean_differences_2


# In[51]:


grader.check("q3_8")


# In[ ]:





# The next cell plots a histogram of the test statistics stored in `mean_differences_2` as well as a red line for `observed_mean_difference_2`.

# In[52]:


bpd.DataFrame().assign(DifferenceInMeans=mean_differences_2).plot(kind='hist')
plt.axvline(observed_mean_difference_2, color='red')


# **Question 3.9.** Assign `p_val_2` to the p-value from this permutation test.

# In[53]:


p_val_2 = (observed_mean_difference_2<=mean_differences_2).mean()
p_val_2


# In[54]:


grader.check("q3_9")


# **Question 3.10.** Assign `q3_10_answer` to the correct conclusion for the permutation test.
# 1. The p-value is less than or equal to the significance level of 0.05. We can thus reject the null hypothesis and conclude that Creative campaigns indeed have a higher average donation amount than Medical campaigns.
# 2. The p-value is less than or equal to the significance level of 0.05. We thus fail to reject the null hypothesis and conclude that we do not have sufficient evidence to determine whether campaigns from these categories follow different distributions.
# 3. The p-value is more than the significance level of 0.05. We thus fail to reject the null hypothesis and conclude that we do not have sufficient evidence to determine that campaigns from these categories follow different distributions.
# 4. The p-value is more than the significance level of 0.05. We can thus reject the null hypothesis and conclude that Creative campaigns indeed have a higher average donation amount than Medical campaigns.

# In[55]:


q3_10_answer = 3
q3_10_answer


# In[56]:


grader.check("q3_10")


# Now that we've successfully completed two permutation tests, let's think about what else we could use this framework for. There are so many interesting questions we could ask about this dataset! 
# 
# **Question 3.11.** Which of the following questions could be answered by running a permutation (A/B) test? Assign `q3_11_answer` to a list containing the numbers for **all** of the following questions that could be answered with an A/B test. Here, "highly successful" and "highly unsuccessful" refer to the definitions given in Question 1.4.
# 
# 1. Does categorizing a campaign as 'Competition' cause it to earn more money than categorizing it as 'Sports'?
# 2. Are campaigns from New York and California more lucrative than campaigns from New Mexico and Iowa?
# 3. Do campaigns with at least 2000 characters in the text description tend to make more money than campaigns with less than 2000 characters?
# 4. Does adding more words to a GoFundMe campaign description cause it to earn more money?
# 5. Do "highly successful" campaigns have more Facebook shares than "highly unsuccessful" campaigns? 
# 6. Are the "highly successful" campaigns distributed uniformly across the different U.S. states/territories?

# In[57]:


q3_11_answer = [1,2,3,5]
q3_11_answer


# In[58]:


grader.check("q3_11")


# <a id='section4'></a>
# # 4. The Power of Words
# In a GoFundMe campaign, fundraiser organizers can write text descriptions about their campaign and give more context to possible donors. Since fundraisers may be shared widely through social media, the description is useful to introduce the person or organization who will benefit from the fundraiser, how the money will be used, why the cause is so important, etc. Run the cell below to see a randomly selected text description from our dataset.

# In[59]:


random_index=np.random.choice(np.arange(gfm.shape[0]))
print(gfm.get('Text').iloc[random_index])


# In this part of the project, we will use A/B testing to analyze whether the length of the description has any effect on the success of a campaign and whether specific words are more impactful.
# 
# We'll start by considering the length of the description. In order to do an A/B test, we need to break the dataset of campaigns into two groups: one representing campaigns with longer text descriptions and the other representing campaigns with shorter text descriptions. Where should the boundary between the two groups be? 
# 
# We could decide arbitrarily (say, 2000 characters is considered long) but as good data scientists, let's use the data we have to determine this cutoff point. Let's try to split the dataset into two equally sized groups. 
# 
# **Question 4.1.** Set `cutoff` to be the smallest number of characters for which we will consider a text description to be long, if we want the groups to be about the same size. 
# 
# *Hint*: We may not be able to split the dataset into groups of the exact same size. What is the best we can do? Use a measure of center.

# In[20]:


med= np.percentile(gfm.get('Num_Chars'), 50)
low= median= np.percentile(gfm.get('Num_Chars'), 25)
high= np.percentile(gfm.get('Num_Chars'), 75)
mean= gfm.get('Num_Chars').mean()
std= np.std(gfm.get('Num_Chars'))
cutoff 
bpd.DataFrame().assign(num_chars=gfm.get('Num_Chars')).plot(kind='hist', density=True, bins= [0,1000,2000,3000,4000,5000,6000,7000,8000])
plt.axvline(med, color='red')
plt.axvline(low, color='blue')
plt.axvline(high, color='green')
med, low, high


# In[61]:


grader.check("q4_1")


# Now define a long description to be a description where the number of characters is ***at least*** as long as `cutoff` and a short description to be a description where the number of characters is less than `cutoff`. 
# 
# **Question 4.2.** Calculate the difference in means of the two groups (long minus short) and assign the result to `observed_mean_difference_length`.

# In[62]:


long_des= gfm[gfm.get('Num_Chars')>= cutoff]
short_des= gfm[gfm.get('Num_Chars')<cutoff]
observed_mean_difference_length = long_des.get('Amount_Raised').mean()-short_des.get('Amount_Raised').mean()
observed_mean_difference_length


# In[63]:


grader.check("q4_2")


# In[ ]:





# Now, we can run a permutation test to help us decide between two hypotheses: 
# 
# **Null Hypothesis**: In the population of GoFundMe campaigns, the distribution of the amount raised is the same for campaigns with long text descriptions and short text descriptions.
# 
# **Alternative Hypothesis**: In the population of GoFundMe campaigns, campaigns with long text descriptions tend to raise more money than campaigns with short text descriptions.
# 
# **Question 4.3.** Calculate 1,000 simulated values of the test statistic and return them as an array called `mean_differences_length`.
# 
# *Hint*: You may use the same framework from the previous A/B tests to help you with this question.

# In[64]:


mean_differences_length = np.array([])
n_repet= 1000
for i in np.arange(n_repet):
    new = np.random.permutation(gfm.get('Num_Chars'))
    shuf_gfm= gfm.assign(Shuf_Num_Chars= new)
    long= shuf_gfm[shuf_gfm.get('Shuf_Num_Chars')>= cutoff]
    short = shuf_gfm[shuf_gfm.get('Shuf_Num_Chars')<cutoff]
    difference= long.get('Amount_Raised').mean()-short.get('Amount_Raised').mean()
    mean_differences_length= np.append(mean_differences_length,difference)
mean_differences_length


# In[ ]:





# In[65]:


grader.check("q4_3")


# In[ ]:





# The next cell plots a histogram of the test statistics stored in `mean_differences_length` as well as a red line for `observed_mean_difference_length`.

# In[66]:


bpd.DataFrame().assign(DifferenceInMeans=mean_differences_length).plot(kind='hist')
plt.axvline(observed_mean_difference_length, color='red')


# If the results of your test aren't obvious from looking at the histogram above, calculate a p-value below.

# In[67]:


(observed_mean_difference_length<=mean_differences_length).mean()


# **Question 4.4.** Do longer campaign descriptions seem to raise more money, according to the results of our A/B test? Set `q4_4_answer` to your answer.
# 1. Yes
# 2. No

# In[68]:


q4_4_answer = 2
q4_4_answer


# In[69]:


grader.check("q4_4")


# Next, we can run some text analysis to determine whether specific words in the `Text` are associated with a higher or lower `Average_Donation_Amount`. More specifically, do donors find specific words found in the description of a campaign impactful? For example, we might conjecture that words relating to family like "father", "daughter", "son", etc. are associated with higher average donation amounts. 
# 
# Which *three* words do you think are impactful to donors? Store any three lowercase words you think are linked to higher average donations in a list called `impactful_words`. 

# In[70]:


impactful_words = ["mother","daughter", "wife" ]
impactful_words


# To determine whether these words actually are associated with higher average donations, we will run a permutation test with the following hypotheses:
# 
# **Null Hypothesis**: In the population of GoFundMe campaigns, the distribution of average donation amount is the same for campaigns whose text description contains *at least one* of the `impactful_words` and campaigns whose text description contains *none* of the `impactful_words`.
# 
# **Alternative Hypothesis**: In the population of GoFundMe campaigns, donors on average tend to donate more to campaigns whose text description contains *at least one* of the `impactful_words` than to campaigns whose text description contains *none* of the `impactful_words`.
# 
# As with our other A/B tests, we'll use the difference in means as our test statistic (impactful minus not impactful).

# **Question 4.5.** Create a new dataframe called `gfm_impactful`, which contains the information from `gfm` plus a column named `Has_Impactful` which is True/False depending whether the campaign contains at least one impactful word or not.

# In[71]:


def contain_words(string):
    if 'mother' in string:
        return True
    if "daughter" in string:
         return True
    if "wife" in string:
        return True
    else:
        return False


# In[72]:


gfm_impactful = gfm.assign(Has_Impactful= gfm.get('Text').apply(contain_words))
gfm_impactful


# In[73]:


grader.check("q4_5")


# In[ ]:





# **Question 4.6.** Now, run the permutation test. Calculate the observed values of the test statistic and store it in `observed_mean_difference_impactful`. Calculate 1,000 simulated values of the test statistic and store them in an array called `mean_differences_impactful`.
# 
# *Hint*: You may use the same framework from the previous A/B tests to help you with this question. Consider writing more functions if you think that would be helpful.

# In[74]:


Is_impact= gfm_impactful[gfm_impactful.get('Has_Impactful')==True]
No_impact= gfm_impactful[gfm_impactful.get('Has_Impactful')==False]
observed_mean_difference_impactful = Is_impact.get('Average_Donation_Amount').mean()- No_impact.get('Average_Donation_Amount').mean()
observed_mean_difference_impactful


# In[75]:


mean_differences_impactful = np.array([])
nrep= 1000
for i in np.arange(nrep):
    new_impact= np.random.permutation(gfm_impactful.get('Has_Impactful'))
    shuf_table= gfm_impactful.assign(Shuf_Has_Impactful=new_impact)
    Has_imp= shuf_table[shuf_table.get('Shuf_Has_Impactful')== True]
    No_imp= shuf_table[shuf_table.get('Shuf_Has_Impactful')== False]
    difference_mean= Has_imp.get('Average_Donation_Amount').mean()-No_imp.get('Average_Donation_Amount').mean()
    mean_differences_impactful= np.append(mean_differences_impactful,difference_mean )
mean_differences_impactful


# In[76]:


grader.check("q4_6")


# In[ ]:





# The next cell plots a histogram of the test statistics stored in `mean_differences_impactful` as well as a red line for `observed_mean_difference_impactful`.

# In[77]:


bpd.DataFrame().assign(DifferenceInMeans=mean_differences_impactful).plot(kind='hist')
plt.axvline(observed_mean_difference_impactful, color='red')


# If the results of your test aren't obvious from looking at the histogram above, calculate a p-value below.

# In[78]:


# Calculate a p-value here if you need to.
(observed_mean_difference_impactful<=mean_differences_impactful).mean()


# Now let's try the same thing on a different set of words to see if we get different results.
# 
# **Question 4.7.** To make it easier to try the same process on a different set of words, create a function called `p_value_impact` that takes as input a list of words (which can contain any number of words, as long as it's at least one) and returns the p-value from a permutation test with that list of words. Use 1,000 permutations in your permutation test, and compare campaigns that contain at least one word in the input list versus campaigns with no words from the input list.

# In[ ]:





# In[79]:


def p_value_impact(word_list):
    differences_impactful_avg = np.array([])
    def listinstring(string):
        for i in word_list:
            if i in string:
                return True
        return False
    gfm2= gfm.assign(impact= gfm.get('Text').apply(listinstring))
    boop= gfm2[gfm2.get('impact')==True].get('Average_Donation_Amount').mean()-gfm2[gfm2.get('impact')==False].get('Average_Donation_Amount').mean()
    for i in np.arange(1000):
        new_gfm2_col= np.random.permutation(gfm2.get('impact'))
        new_gfm2= gfm2.assign(shuffeled_impact= new_gfm2_col)
        gfm_impact= new_gfm2[new_gfm2.get('shuffeled_impact')==True]
        gfm_no_impact= new_gfm2[new_gfm2.get('shuffeled_impact')== False]
        mean_dif= gfm_impact.get('Average_Donation_Amount').mean()-gfm_no_impact.get('Average_Donation_Amount').mean()
        differences_impactful_avg= np.append(differences_impactful_avg,mean_dif )
    return (boop<=differences_impactful_avg).mean()
p_value_impact(impactful_words)


# In[ ]:





# In[80]:


#Don't change this cell, but do run it. This is used to test if your code is correct.
mother_test = p_value_impact(['mother'])
mother_test


# In[81]:


grader.check("q4_7")


# Now try out your function on some word lists. Here are a couple, but try out your own ideas, too!

# In[82]:


p_value_impact(['hope', 'god', 'tragic'])


# In[83]:


p_value_impact(['covid', 'family', 'surgery', 'school'])


# In[84]:


#Try some of your own word lists, of any length.
p_value_impact(["emergency", "hospital", "funeral"])


# Did the results from your permutation test surprise you? Were the words you chose actually impactful or not? If not, can you find a different set of words that is impactful, or are you getting the sense that words in the text description don't actually matter all that much? How about if you look for *all* of the words in the word list instead of *at least one*. Does that change things? 
# 
# This is just a glimpse of some of the insights you can gain through text analysis!

# <a id='section5'></a>
# # 5. Average Amount Raised

# For this part of the project, we want to learn more about how much money, on average, is raised by GoFundMe fundraisers.
# 
# Well, easy. We can just take the average of our dataset. Right?
# 
# Wrong.
# 
# Our dataset is only a sample of 837 fundraisers. While 837 might sound like a lot of fundraisers, GoFundMe has a huge number of campaigns, with more being created each day. According to their [website](https://www.gofundme.com/), "over 10,000 people start a GoFundMe every day."
# 
# So how do we find the average amount raised from ALL GoFundMe fundraisers, if we only have a small sample? We'll never be able to find this number exactly, but we can estimate it. How?
# 
# You guessed it: we can use bootstrapping!

# **Question 5.1.** Create a function called `bootstrap` which resamples from `gfm` and computes the average Amount Raised for each resample. The function should take as input a parameter `n` and create `n` bootstrap resamples. It should return an array of length `n` containing the average Amount Raised for each resample. 

# In[85]:


def bootstrap(n):
    average_raised_mean= np.array([])
    for i in np.arange(n):
        resample = gfm.sample(gfm.shape[0],replace=True)
        resample_mean= resample.get('Amount_Raised').mean()
        average_raised_mean = np.append(average_raised_mean,resample_mean)
    return average_raised_mean


# In[86]:


grader.check("q5_1")


# In[ ]:





# **Question 5.2.** Call your function `bootstrap` to generate an array called `boot_data` containing 1,000 bootstrapped averages. Compute the upper and lower bounds of the 95% bootstrapped confidence interval for this data. Store these values in variables called `upper_bound` and `lower_bound`.
# 
# *Hint*: Use the `np.percentile()` method. You may want to plot a histogram of `boot_data` to make sure your bounds are reasonable.

# In[87]:


boot_data = bootstrap(1000)
lower_bound = np.percentile(boot_data,2.5)
upper_bound = np.percentile(boot_data,97.5)
print(lower_bound, upper_bound)


# In[88]:


grader.check("q5_2")


# In[89]:


bpd.DataFrame().assign(Estimated_Mean = boot_data).plot(kind = 'hist')
plt.axvline(lower_bound, color='red')
plt.axvline(upper_bound, color='blue')


# **Question 5.3.**  Your very rich and very generous friend decides that they want to randomly pick a fundraiser and double its earnings by donating the same amount that has been raised so far. They told you that they have exactly the same amount of money as the upper bound of your 95% confidence interval.
# 
# Your friend is elated as they announce that they can afford to double the earnings of at least 95 percent of the GoFundMe fundraisers, so they'll be able to accomplish their goal with at least 95 percent probability.
# 
# Is your friend right? Can they afford to double at least 95% of all fundraisers? Set `q5_3_answer` to True or False.

# In[90]:


q5_3_answer = False
q5_3_answer


# In[91]:


grader.check("q5_3")


# <a id='section6'></a>
# # 6. Sharing the Wealth

# This is the last section of the Final Project, so congrats! You're almost there! 
# 
# More than ever, especially in these current times, we've been heavily relying on the use of technology to keep us together virtually when we are physically restricted. Here, we will be analyzing the possible effects that social media platforms, in this case Facebook, have on the visibility of social causes such as GoFundMe fundraisers. If you've used Facebook before, more likely than not, you have seen a GoFundMe fundraiser being shared by users to spread awareness. 
# 
# In this section, we're examining the possible underlying effects that social media platforms such as Facebook can have on the performance of fundraisers by spreading awareness through shares. To examine the relationship between Facebook shares and amount raised, we use regression, and we will even predictions about how much money a fundraiser could earn based on the number of Facebook shares it has.

# **Question 6.1.** Create a scatter plot of the number of Facebook shares each fundraiser received versus the total amount it raised.

# <!-- BEGIN QUESTION -->
# 
# <!--
# BEGIN QUESTION
# name: q6_1
# manual: true
# -->

# In[16]:


gfm.plot(kind = 'scatter', x = 'FB_Shares', y = 'Amount_Raised')


# <!-- END QUESTION -->
# 
# 
# 
# From the scatter plot you've generated above, we can see that there are definitely some outlier values for both the number of Facebook shares each fundraiser received and the amount of money it raised. Let's remove these outliers from our dataset to get a more representative view of our data.
# 
# Let's establish that any fundraiser with **more than four thousand Facebook shares** or raising **more than one million dollars** will be considered an outlier. We've settled upon these values somwhat arbitrarily. Our goal is to look at the majority of fundraisers and not have our results be too skewed by the exceptional fundraisers. These values (4,000 Facebook shares and $1,000,000) allow us to go keep the bulk of our data while discarding the ones that could lead to misleading conclusions.

# **Question 6.2.**  Starting with `gfm`, create a new dataframe called `without_outliers` that excludes any fundraisers meeting this criteria. As in the previous question, make a scatter plot of the number of Facebook shares versus the amount raised for each fundraiser in `without_outliers`.

# In[18]:


without_outliers_fb = gfm[gfm.get('FB_Shares')<=4000]
without_outliers= without_outliers_fb[without_outliers_fb.get('Amount_Raised')<=1_000_000]
without_outliers


# In[ ]:





# In[19]:


without_outliers.plot(kind = 'scatter', x = 'FB_Shares', y = 'Amount_Raised')


# In[95]:


grader.check("q6_2")


# **Question 6.3.** If you take a closer look at this scatter plot, you'll see that the some of the data points fall into vertical stripes, particularly in the range of around 1,000 to 2,000 Facebook Shares. Which of the following answers below best explains why these vertical lines appear? Store your answer (1 through 4) as `striped_data`.

# 1. Data points in vertical lines are fundraisers from the same category.
# 2. Data points in vertical lines are fundraisers with the same goal.
# 3. The amount raised for some fundraisers has been rounded to certain values.
# 4. The number of Facebook shares for some fundraisers has been rounded to certain values.

# In[96]:


striped_data = 4
striped_data


# In[97]:


grader.check("q6_3")


# **Question 6.4.** Now we want to create a regression line for the data in our scatter plot without outliers, so we can make predictions about the amount of money a campaign will raise based off the number of Facebook shares it has. To do so, we first need to standardize these values. Complete the function `standardize` which should take in a Series of numbers from our dataframe and output a Series of their standardized values.

# In[98]:


def standardize(data):
    return (data - data.mean())/np.std(data)


# In[99]:


grader.check("q6_4")


# In[ ]:





# In[100]:


type(standardize(without_outliers.get('Amount_Raised')))


# **Question 6.5.** Now that we have a function to standardize a list of numbers, let's put it to use! Create a new dataframe called `standardized` that has the same data as `without_outliers`, plus two new columns, `Standardized_FB_Shares` and `Standardized_Amount_Raised`, containing the standardized values of the number of Facebook Shares and the amount raised of each fundraiser respectively.

# In[101]:


standardized = without_outliers.assign(Standardized_FB_Shares= standardize(without_outliers.get('FB_Shares')),Standardized_Amount_Raised= standardize(without_outliers.get('Amount_Raised')) ) 
standardized


# In[102]:


grader.check("q6_5")


# **Question 6.6.** Find the correlation coefficient, `r`, for `Amount_Raised` and `FB_Shares` using the standardized units you calculated in the previous question.

# In[103]:


r = (standardized.get('Standardized_FB_Shares')*standardized.get('Standardized_Amount_Raised')).mean()
r


# In[104]:


grader.check("q6_6")


# **Question 6.7.** Based off the correlation coefficient coefficient you've calculated, which of the following best describes the relationship between Facebook Shares and Amount Raised?
# 
# 1. Strongly negatively correlated
# 2. Weakly negatively correlated
# 3. Not correlated
# 4. Weakly positively correlated
# 5. Strongly positively correlated
# 
# For the sake of this question, assume that a magnitude of 0.70 or greater suggests a strong correlation. Save your answer as `q6_7_answer` below.

# In[105]:


q6_7_answer = 4
q6_7_answer


# In[106]:


grader.check("q6_7")


# **Question 6.8.** With our correlation coefficient, we can now construct our regression line! Calculate the slope (`m`) and the intercept (`b`) of our regression line.

# In[107]:


m = r * np.std(without_outliers.get('Amount_Raised'))/np.std(without_outliers.get('FB_Shares'))
b = without_outliers.get('Amount_Raised').mean()- m*without_outliers.get('FB_Shares').mean()
m, b


# In[108]:


grader.check("q6_8")


# Now that we've created a formula for our regression line, of course we're going to want to see how it looks on our scatter plot! Run the cell below to see your regression line plotted onto the scatter plot.

# In[109]:


without_outliers.plot(kind = 'scatter', x = 'FB_Shares', y = 'Amount_Raised')
x_values = np.arange(0, 4001)
y_values = x_values * m + b
plt.scatter(x_values, y_values, color='red', s = 0.7)


# **Question 6.9.** Knowing the slope and intercept of our regression line, we can go ahead and make predictions about the amount of money we expect a campaign to raise based on the number of Facebook shares it receives. About how much money would you predict a fundraiser would generate if it received 2,750 shares on Facebook? Save the result, rounded to the nearest cent, as `predicted_funds`.

# In[110]:


predicted_funds = m * 2750  + b
predicted_funds


# In[111]:


grader.check("q6_9")


# **Question 6.10.** Time to reminisce all the way back to the very first few lectures of DSC 10, when we were discussing correlation and causation. From what you've uncovered in this project, can you say that having more Facebook shares causes  a fundraiser to earn more money? Set `q6_10_answer` to your answer.
# 1. Yes
# 2. No

# In[112]:


q6_10_answer = 2
q6_10_answer


# In[113]:


grader.check("q6_10")


# # Finish Line: Almost there, but make sure to follow the steps below to submit!

# Big congratulations! You've completed the Final Project! To submit your assignment:
# 
# 1. Select `Kernel -> Restart & Run All` to ensure that you have executed all cells, including the test cells.
# 2. Read through the notebook to make sure everything is fine and all tests passed. If you fail a test here that used to pass, you probably changed that variable sometime later. Check through your code and make sure to use new variable names rather than overwriting variables that are used in the tests.
# 3. Run the cell below to run all tests, and make sure that they all pass.
# 4. Download your notebook using `File -> Download as -> Notebook (.ipynb)`, then upload your notebook to Gradescope.
# 
# Remember, the tests here and on Gradescope just check the format of your answers. We will run correctness tests after the assignment's due date has passed.
# 
# **Long Simulations.** If any of your cells are taking more than five minutes to run, you are probably doing something wrong. You can sometimes speed things up by making sure you have a table of only the rows and columns you need to do your analysis, which should be defined outside the for loop of your simulation. Make sure your table is as small as possible in both rows and columns. When possible, try to avoid using additional for loops and queries inside a simulation, and see if a faster method, like a numpy method or groupby, could be used instead. 
# 
# **Total Runtime.** Please test the total run time using `Kernel -> Restart & Run All`. If it takes more than thirty minutes, you are probably going to have problem with gradescope. Please make sure your runtime is under 30 minutes. 

# In[114]:


grader.check_all()


# In[ ]:





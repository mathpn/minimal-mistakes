---
title: "What are **NOT** _p-values?_"
header:
  overlay_image: /assets/images/default_overlay.jpg
  show_overlay_excerpt: false
categories:
  - Blog
tags:
  - Statistics
excerpt: "What exactly is a p-value? Let's define the p-value and then look at what it is not."

---

Over the last 100 years, _p-values_ have become increasingly common in many scientific fields, after its development by Ronald Fisher in 1920. Now, on the 100th anniversary of the all-famous _p-value_, its use is being questioned due to irreproducible research, _p-hacking_ practices and various misunderstandings about its true meaning.

Reporting only _p-values_ is becoming less acceptable, and for good reason. Scientists should also report full data distribution, confidence intervals and details about the statistical test used and its assumptions. Those who continue to report only that P < 0.05 will face more and more questions from journals and reviewers. All this is good for science and open science. Still, _p-values_ are very useful and will not disappear - at least not for now.

But what exactly is a _p-value_? Many scientists have never thought about this question explicitly. So, let's define the _p-value_ and then look at what it is **not**. Defining what something is not is a great wat to remove misconceptions that may already be lurking in our heads.

The American Statistical Association defines the _p-value_ as:
> [...] The probability under a specified statistical model that a statistical summary of the data (e.g., the sample mean difference between two compared groups) would be equal to or more extreme than its observed value.

<cite>American Statistical Association</cite> --- [ASA Statement on Statistical Significance and P-Values](https://amstat.tandfonline.com/doi/full/10.1080/00031305.2016.1154108)
{: .small}

Most often, the _specified statistical model_ is the null hypothesis: H0 or the hypothesis that group means (or another statistical summary, such as median) are **not different**. Or that the correlation between two continuous variables is **not different** from 0. I guess 99% of the times we encounter _p-values_ it's under these circumstances.

So, the _p-value_ is the probability that, **given that H0 is true** (this is _extremely important_), the results would come up with at least the difference observed.

Let's look at what _p-values_ are **not**:

## 1) The probability that the results are due to chance
Probably the most common misconception regarding _p-values_, it's easy to see why we fall into this statistical trap. Assuming that there is _no difference_ between the two groups, all the observed difference is due to chance. The problem here is that _p-value_ calculations **assume** that **every deviation from H0 is due to chance**. Thus, it cannot compute a probability of something it assumes to be true.

_P-values_ tell us the probability that the observed results **_would_ come up due to chance alone _assuming H0 to be true_**, and **not** the chance that the observed results are due to chance, precisely because **we don't know if H0 is true or not**. Pause and think about the difference between these two statements. If the difference is not clear, it may become clearer with the next topic.

## 2) The probability that H1 is true
This is a bit tricky to understand. H1 is the alternative hypothesis, in contrast to H0. Almost always, H1 states that the groups are different or that an estimator is different from zero.
However, **_p-values_ tell us nothing about H1.** They tell us $P(observed\ difference | H0)$, or the probability of the observed difference _given that H0 is true_. Since H0 and H1 are complementary hypotheses, $P(H0) + P(H1) = 1$. Thus, $P(H1) = 1 - P(H0)$.

However, **we do not know P(H0)** since we assume H0 to be true. Let's use an example.

Let's assume we are a patient doing blood tests for a disease called _statisticosis_. We know that, if you're ill, the test returns a positive result 99% of the time. Also, if you're not ill, it returns a negative result 98% of the time. 1% of the population is estimated to have statisticosis. Let's assume a population of 1 million people and build a table. First, there are 990,000 people (99% of the population) without the disease and 10,000 people (1%) with the disease. Of those that are ill, 9,900 (99% of the ill) will get a positive result at the test, while 9,900 people that are not ill will also get a positive result (2% of those who are not ill).

|  | Ill | Not ill |
|:--------|:-------:|--------:|
| **+ test**   | 9,900   | 19,800   |
| **- test**  | 100   | 970,200   |

Here, H0 is that we're not sick, while H1 is that we are sick. The probability of getting a positive result **given that we are not sick** is.

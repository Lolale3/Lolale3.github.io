## Featured

### Gatekeeper — a confidence-gating layer for LLM extraction

An open-source Python library that decides, field by field, when to trust an LLM's extraction and when to escalate it to a human.

Most extraction systems hand back an answer and leave you guessing whether it's right. Gatekeeper scores each field's reliability, calibrates that score into a real probability, and applies a cost-aware threshold — so the model escalates exactly the cases it shouldn't be trusted on. Full pipeline: extract → signal → calibrate → decide → measure → improve.

Evaluated on real invoice OCR data with a local qwen2.5 model — **~76% precision at 50% coverage, AURC 0.23, ECE ~0.03.** The ablation was the interesting part: self-consistency did nearly all the work (AURC 0.34 → 0.23), while a two-temperature signal I expected to help measured as completely redundant. I kept that result in the README rather than quietly dropping it.

Python, zero required dependencies, 67 tests, four swappable interfaces.

![Risk-coverage curve](https://lolale3.github.io/images/real_risk_coverage.png?raw=true)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/LLM-white?logo=LLM)](#) [![](https://img.shields.io/badge/Calibration-white?logo=Calibration)](#) [![](https://img.shields.io/badge/Selective_Prediction-white?logo=Selective_Prediction)](#) [![](https://img.shields.io/badge/Evaluation-white?logo=Evaluation)](#) [![](https://img.shields.io/badge/Streamlit-white?logo=Streamlit)](#)

[View the repo on GitHub](https://github.com/Lolale3/gatekeeper)

---

## Selected projects

### Credit Risk Modeling and Scorecard Development

Predicted probability of default on Lending Club consumer loan data (2007–2014). Used chi-square and ANOVA F-tests for feature selection, re-engineered features with Weight-of-Evidence, applied Logistic Regression to model default probability, and built a credit scorecard.

The outcome that mattered: setting the loan cut-off deliberately, balancing credit score against approval/rejection rates rather than accepting a default threshold — which brought the rejection threshold from 50% down to 7.6%.

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/Logistic_Regression-white?logo=Logistic_Regression)](#) [![](https://img.shields.io/badge/Weight_of_Evidence-white?logo=Weight_of_Evidence)](#) [![](https://img.shields.io/badge/Statistics-white?logo=Statistics)](#) [![](https://img.shields.io/badge/sklearn-white?logo=scikit-learn)](#)

[View code on GitHub](https://github.com/Lolale3/Credit_Risk_Modeling)

---

### Customer Churn Prediction and Prevention using Survival Analysis

Built a churn model on 7,000 telecom customers, then went past classification into survival analysis — measuring not just *who* churns but *when*, and how specific contract changes shift the point at which a customer hits a 50% chance of leaving.

Kaplan-Meier for the survival curves, Cox proportional hazards for the drivers.

![Kaplan-Meier](https://lolale3.github.io/images/Kaplan-Meier.png?raw=true)![CoxPH](https://lolale3.github.io/images/CoxPH.png?raw=true)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/Survival_Analysis-white?logo=Survival_Analysis)](#) [![](https://img.shields.io/badge/Kaplan_Meier-white?logo=Kaplan_Meier)](#) [![](https://img.shields.io/badge/CoxPH-white?logo=CoxPH)](#) [![](https://img.shields.io/badge/Logistic_Regression-white?logo=Logistic_Regression)](#)

[View code on GitHub](https://github.com/Lolale3/Customer-Churn-Analysis)

---

### Sales Analytics & Customer Segmentation

Analyzed $10.1M in sales data (2003–2005), then used RFM analysis to segment customers into four behavioral groups — and quantified what each was actually worth to go after.

The output wasn't a cluster plot. It was a ranked list of moves with revenue attached: **$2.37M in opportunity across four segments** — $1.29M from reactivating steady mid-tier buyers, $735K from bundling for price-sensitive infrequent ones, $185K from winning back inactive big-ticket customers, $162K from upselling the loyal core.

![RFM customer segments](https://lolale3.github.io/images/cluster.png?raw=true)

Underneath it, a 3-quarter rolling YoY growth analysis by country surfaced where momentum was actually building versus where it was quietly dying — the UK up 38.3% in 2005 Q2, Singapore falling off a cliff, and the USA decelerating from 26.5% to 8.1%, an early saturation signal.

![Rolling YoY growth by country](https://lolale3.github.io/images/YoY_Sales_Growth_sales.jpg?raw=true)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/RFM-white?logo=RFM)](#) [![](https://img.shields.io/badge/Segmentation-white?logo=Segmentation)](#) [![](https://img.shields.io/badge/Time_Series-white?logo=Time_Series)](#) [![](https://img.shields.io/badge/Cohort_Analysis-white?logo=Cohort_Analysis)](#) [![](https://img.shields.io/badge/sklearn-white?logo=scikit-learn)](#)

[View code](YOUR_SALES_ANALYTICS_LINK)

---

### Disaster Tweet Classification — NLP with BERT

Separating genuine disaster signals from noise on 7,000+ tweets, as part of Data Science Olympians 2023 for Women in Big Data.

Built the full ladder rather than jumping straight to the transformer: TF-IDF and lemmatization baselines, then Logistic Regression, XGBoost, and MLP, then fine-tuned BERT. The interesting part was evaluation — choosing the metric based on the real-world cost of a false negative, not chasing accuracy.

![Disaster tweets](https://lolale3.github.io/images/Diasater_tweets.png?raw=true)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/BERT-white?logo=BERT)](#) [![](https://img.shields.io/badge/NLP-white?logo=NLP)](#) [![](https://img.shields.io/badge/XGBoost-white?logo=XGBoost)](#) [![](https://img.shields.io/badge/sklearn-white?logo=scikit-learn)](#)

[View code on Colab](https://colab.research.google.com/drive/1A3k4fnIB35JXjbteRTUoQs-RqIkvhcZf?usp=sharing)

---

## Earlier work

### Apparel Recommendation using Classification and Clustering

Top-5 visual similarity recommendations over Fashion-MNIST (70,000 images, 10 apparel classes). SVM to classify, K-means to cluster, then nearest-neighbour retrieval within a cluster — a content-based recommender built from classical components.

![Apparel recommendation](https://lolale3.github.io/images/apparel_rec.png?raw=true)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/SVM-white?logo=SVM)](#) [![](https://img.shields.io/badge/KMEANS-white?logo=KMEANS)](#) [![](https://img.shields.io/badge/Numpy-white?logo=Numpy)](#)

[View code on Colab](https://colab.research.google.com/drive/1LiZrpuE9ECFyLGaG0gwjRgKZxBK89n-m?usp=sharing)

---

### Healthcare Marketing Insights Dashboard

A Power BI dashboard tracking the metrics a marketing team actually gets judged on — ROAS, CAC, conversion rate, and cost per lead — so channel spend can be evaluated and reallocated rather than guessed at.

Cleaned and modeled a year of operational data in Python underneath it, supporting agent performance tracking and funnel analysis.

![Healthcare Marketing Dashboard](https://lolale3.github.io/images/Healthcare_dashboard.png?raw=true)

[![](https://img.shields.io/badge/PowerBI-white?logo=PowerBI)](#) [![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/Marketing_Analytics-white?logo=Marketing_Analytics)](#) [![](https://img.shields.io/badge/ROAS-white?logo=ROAS)](#) [![](https://img.shields.io/badge/Funnel_Analysis-white?logo=Funnel_Analysis)](#)

[View dashboard](https://drive.google.com/file/d/11T06vUqNQt5nfHWR77VJE8nIiFQkf95L/view?usp=drive_link)

---

Page template forked from [evanca](https://github.com/evanca/quick-portfolio)

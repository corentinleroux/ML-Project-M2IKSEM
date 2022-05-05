<p align="center">
  <img src="https://study-eu.s3.amazonaws.com/uploads/university/universit--paris-1-panth-on-sorbonne-479-logo.png">
</p>
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/v1/5a37d32fbce1765b74b2f6b2/1528477847075-CUX2PFJFGA36B1NBQR2D/divorce.png?format=750w" width="150">
</p>



# Machine Learning Project - M2IKSEM

Project realized for the Machine Learning course of the Master 2 Informatique MIAGE at the University Paris-1 Panthéon-Sorbonne. We will use Ensemble Learning. Ensemble learning means to train multiple base learners and combine their predictions in an optimal way into a single better output. Our group is composed of :

- MEURIC Camille
- BOUDRAA Inès
- LE ROUX Corentin
- CHAKHCHOUKH Lina
  

## Technologies

![Python](https://img.shields.io/badge/Python-3.X.X-success)  ![Jupyter](https://img.shields.io/badge/Jupiter%20Notebook-6.0.1-blue) 

![scikit](https://img.shields.io/badge/scikit--learn-0.21-orange) ![pandas](https://img.shields.io/badge/pandas-0.25-orange) ![numpy](https://img.shields.io/badge/numpy-1.17-orange) ![seaborn](https://img.shields.io/badge/seaborn-0.9-orange)


## Installation guide

- If you dont have Python : [Install it](https://www.python.org/downloads/)
-  `$ sudo apt update -y`
-  `$ sudo apt install python3.9` or `brew install python@3.9`
-  `$ pip3 install ipython`
-  `$ pip3 install jupyter`
-  `$ pip3 install pandas`
-  `$ pip3 install scikit-learn`
-  `conda install -c conda-forge missingno` or `conda install -c conda-forge/label/gcc7 missingno`
- `gh repo clone corentinleroux/ML-Project-M2IKSEM`

        
## Run the notebooks

Use `jupyter notebook` to run the interface for Notebooks.  

## Run the website

- Go to website folder and run `flask run` 
- For the debug mode, run `FLASK_APP=app.py FLASK_ENV=development flask run`

## Dataset

We took the dataset on Kaggle : [Kaggle](https://www.kaggle.com/csafrit2/predicting-divorce).

<details>
  <summary>Click here to see the list of questions used to create the dataset</summary>
 -----
  
Questions are ranked on a scale of 1-5 with 1 being the lowest and 5 being the highest. The last category states if the couple has divorced.

1. If one of us apologizes when our discussion deteriorates, the discussion ends.
  
2. I know we can ignore our differences, even if things get hard sometimes.
  
3. When we need it, we can take our discussions with my spouse from the beginning and correct it.
4.	When I discuss with my spouse, to contact him will eventually work.
5.	The time I spent with my wife is special for us.
6.	We don't have time at home as partners.
7.	We are like two strangers who share the same environment at home rather than family.
8.	I enjoy our holidays with my wife.
9.	I enjoy traveling with my wife.
10.	Most of our goals are common to my spouse.
11.	I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.
12.	My spouse and I have similar values in terms of personal freedom.
13.	My spouse and I have similar sense of entertainment.
14.	Most of our goals for people (children, friends, etc.) are the same.
15.	Our dreams with my spouse are similar and harmonious.
16.	We're compatible with my spouse about what love should be.
17.	We share the same views about being happy in our life with my spouse
18.	My spouse and I have similar ideas about how marriage should be
19.	My spouse and I have similar ideas about how roles should be in marriage
20.	My spouse and I have similar values in trust.
21.	I know exactly what my wife likes.
22.	I know how my spouse wants to be taken care of when she/he sick.
23.	I know my spouse's favorite food.
24.	I can tell you what kind of stress my spouse is facing in her/his life.
25.	I have knowledge of my spouse's inner world.
26.	I know my spouse's basic anxieties.
27.	I know what my spouse's current sources of stress are.
28.	I know my spouse's hopes and wishes.
29.	I know my spouse very well.
30.	I know my spouse's friends and their social relationships.
31.	I feel aggressive when I argue with my spouse.
32.	When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’ .
33.	I can use negative statements about my spouse's personality during our discussions.
34.	I can use offensive expressions during our discussions.
35.	I can insult my spouse during our discussions.
36.	I can be humiliating when we discussions.
37.	My discussion with my spouse is not calm.
38.	I hate my spouse's way of open a subject.
39.	Our discussions often occur suddenly.
40.	We're just starting a discussion before I know what's going on.
41.	When I talk to my spouse about something, my calm suddenly breaks.
42.	When I argue with my spouse, ı only go out and I don't say a word.
43.	I mostly stay silent to calm the environment a little bit.
44.	Sometimes I think it's good for me to leave home for a while.
45.	I'd rather stay silent than discuss with my spouse.
46.	Even if I'm right in the discussion, I stay silent to hurt my spouse.
47.	When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.
48.	I feel right in our discussions.
49.	I have nothing to do with what I've been accused of.
50.	I'm not actually the one who's guilty about what I'm accused of.
51.	I'm not the one who's wrong about problems at home.
52.	I wouldn't hesitate to tell my spouse about her/his inadequacy.
53.	When I discuss, I remind my spouse of her/his inadequacy.
54.	I'm not afraid to tell my spouse about her/his incompetence.
55. Divorce Y/N 
</details>

## References 

About Dataset : 

- [Divorce Prediction Using Correlation Based Feature Selection And Artificial Neural Networks](https://www.researchgate.net/publication/334170931_DIVORCE_PREDICTION_USING_CORRELATION_BASED_FEATURE_SELECTION_AND_ARTIFICIAL_NEURAL_NETWORKS)

About Machine Learning, Ensemble Learning, Bagging, Boosting, Stacking :

- [Ensemble methods: bagging, boosting and stacking](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205)
- [Ensemble Learning: Bagging & Boosting](https://towardsdatascience.com/ensemble-learning-bagging-boosting-3098079e5422)
- [Les méthodes ensemblistes pour algorithmes de machine learning](https://blog.octo.com/les-methodes-ensemblistes-pour-algorithmes-de-machine-learning/)
- [Bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias–variance_tradeoff)

## Choice of method

We decided to try both the **Boosting method** and the **Bagging method** to see which one fit the most our ML Project.  


- **What is variance ?**
> Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data. (*source : [Understanding the Bias-Variance Tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)*)

- **What is bias?**
> Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data. (*source : [Understanding the Bias-Variance Tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)*)

- **How does it work ?**
> <img src="https://miro.medium.com/max/700/1*zTgGBTQIMlASWm5QuS2UpA.jpeg" width="500">
> 
> (*source : [Ensemble Learning: Bagging & Boosting](https://towardsdatascience.com/ensemble-learning-bagging-boosting-3098079e5422)*)

- **What is the difference ?** 
> The choice between the 3 methods depend on what we want to optimize (see the Bias-variance tradeoff). With parrallelism, Bagging aim to decrease variance. It is best suitable for high variance low bias models. On the other hand, with sequential ensemble, Boosting aim to decrease bias and is suitable for low variance high bias models.

## Results

> Some results paragraph and tables

| Classification | Accuracy (%) |
| :---:   | :-: | 
| Bagging | 0.96 |
| Gradient Boosting | 0.96 | 
| Stacking | 0.95 |
| Random Forest | 0.98 | 
| AdaBoost | 0.98 | 
| KNN | 0.98 |

## Demo

### Website 
A demo is available at the following link :  [Youtube - Demonstration](https://youtu.be/dkttPtbD614)

[![Demonstration](https://i.ibb.co/vX2LxSf/Capture-d-cran-2022-05-05-13-50-35.png)](https://youtu.be/dkttPtbD614 "Website Demo")


### Notebooks
A demo is available at the following link :  [Youtube - Demonstration](https://youtu.be/Gebm9YGn4Lg)

[![Demonstration](https://i.ibb.co/jVt0jsK/Capture-d-cran-2022-05-05-13-54-48.png)](https://youtu.be/Gebm9YGn4Lg "Presentation")

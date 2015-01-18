# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

training_documents = [
    '''
Poor sleep in teenage years could be an early warning sign for alcohol problems, illicit drug use and "regretful" sexual behaviour, research suggests.
US scientists found adolescents with bad sleep habits were more likely to engage in risky behaviour in the years to come than those who slept soundly.
They say parents should pay closer attention to teens' sleep schedules.
Other research suggests a good night's sleep is key to making good judgements.
''', '''
Prioritising medical students with lots of friends for flu jabs could help increase the number of healthcare workers protected against the virus, say Lancaster University researchers.
In a study in The Lancet, they calculated that vaccination rates would rise if people with large social networks influenced their peers.
The government wants 75% of healthcare workers to be vaccinated.
At present, only about half of them are vaccinated.
More than 200 medical students at Lancaster University - who are soon to become healthcare workers - gave researchers information on how friendly they were with other students and how much time they spent with them.
''', '''
West Ham football club sponsor and currency broker Alpari has shut its UK arm following the Swiss National Bank's decision to end its capping of the Swiss franc against the euro.
The foreign exchange broker said in a statement that the move had created "exceptional volatility and extreme lack of liquidity".
As a result, the majority of Alpari clients had "sustained losses".
The euro rose 1.2% on Friday to buy 0.9869 Swiss francs.
''', '''
The European Commission has disclosed a preliminary finding that Amazon's tax arrangements in Luxembourg probably constitute "state aid".
The EC's doubts about the arrangement were detailed in a document on Friday.
The EC said that its "preliminary view is that the tax ruling... by Luxembourg in favour of Amazon constitutes state aid."
However, Amazon said it "has received no special tax treatment from Luxembourg".
''', '''
Hi-tech and high fashion have always been closely intertwined.
And these days, being seen with the right gadget or using the right app is as important as knowing that stripes are out and dots are in or that beards are big and bushy this season.
Elsewhere on the show floor, we've seen how companies are putting in a lot of effort to make wearable technology a little more inconspicuous than what's on offer right now.
''', '''
The president of the Academy of Motion Picture Arts says she would like to see more diversity in Oscar nominations, after a row about this year's nominees.
All 20 contenders in the main acting categories are white and there are no female nominees in the directing or writing categories.
The Academy, which picks the contenders, has faced strong criticism.
But Cheryl Boone Isaacs said she was proud of the nominees and that the body was "making strides" towards diversity.
'''
]
# 0: Health, 1: Business, 2: Other
training_labels = [0, 0, 1, 1, 2, 2]

test_documents = [
    '''
Closing your eyes when trying to recall events increases the chances of accuracy, researchers at the University of Surrey suggest.
Scientists tested people's ability to remember details of films showing fake crime scenes.
They hope the studies will help witnesses recall details more accurately when questioned by police.
They say establishing a rapport with the person asking the questions can also help boost memory.
''', '''
BP faces a fine of up to $13.7bn (£9bn) after a US judge ruled that the 2010 Gulf of Mexico oil spill was smaller than initially feared.
His ruling put the spill at 3.2 million barrels - the US government had estimated it at 4.09 million barrels.
It shields the oil giant from what could have been a $17.6bn fine. A final figure is expected later this month.
The case relating to the aftermath of the Deepwater Horizon drilling rig explosion was heard in New Orleans.
''', '''
Wearable tech is staging a vanishing act.
That's not to say that there you can't find any new products here at CES in Las Vegas.
In fact, the show floor is packed with me-too fitness tracking bracelets, cut-price touchscreen smartwatches and strap-on heart rate monitors.
But a select number of exhibitors is attempting to sell products that do not flag the fact the wearer is sporting a gadget.
'''
]
test_labels = [0, 1, 2]

vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
training_vector = vectorizer.fit_transform(training_documents)
print("training documents: %d, unique words: %d" % training_vector.shape)
# training documents: 6, unique words: 210

estimator = LinearSVC()
estimator.fit(training_vector, training_labels)

test_vector = vectorizer.transform(test_documents)
print("test documents: %d, unique words: %d" % test_vector.shape)
# test documents: 3, unique words: 210

predicted_labels = estimator.predict(test_vector)
print("  correct labels: {}".format(test_labels))
print("predicted labels: {}".format(predicted_labels))
print("accuracy_score: {}".format(accuracy_score(test_labels, predicted_labels)))
print(classification_report(test_labels, predicted_labels))
#              precision    recall  f1-score   support
#
#           0       1.00      1.00      1.00         1
#           1       1.00      1.00      1.00         1
#           2       1.00      1.00      1.00         1
#
# avg / total       1.00      1.00      1.00         3
